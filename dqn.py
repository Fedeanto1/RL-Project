"""
DRL Project — DQN
==================
Based on CleanRL's DQN implementation:
  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

Original additions:
  - Configurable hyperparameter dictionary replacing CleanRL's hardcoded values
  - StateVisitationTracker integration for exploration heatmaps
  - Replay buffer ablation (buffer_size=0 = no-replay ablation)
  - Epsilon scheduling variants (configurable eps_start, eps_end, eps_decay_steps)
  - W&B logging of custom metrics (epsilon, buffer fill rate, loss)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from collections import deque

from drl_utils import DEVICE, set_seed, MinAtarEnv, StateVisitationTracker


# ── Q-Network ──────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    CNN Q-network for MinAtar.

    Architecture: Conv(C->16, 3x3) -> ReLU -> Flatten -> Linear(16HW->128)
                  -> ReLU -> Linear(128->n_actions)

    Input : (batch, C, H, W)
    Output: (batch, n_actions)  — one Q-value per action
    """
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        C, H, W = obs_shape #obs_shape is (4, 10, 10) for MinAtar Breakout
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        conv_out = 16 * H * W
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


# ── Replay Buffer ──────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Experience replay buffer — key component of off-policy DQN.

    Stores (obs, action, reward, next_obs, done) transitions and serves
    random mini-batches to break temporal correlation between samples.

    buffer_size=0 triggers the no-replay ablation: only the latest
    transition is used for each update, making DQN behave similarly to
    an on-policy method and exposing the instability that replay prevents.
    """
    def __init__(self, capacity: int, obs_shape):
        self.capacity  = capacity
        self.no_replay = (capacity == 0)
        maxlen = 1 if self.no_replay else capacity  #If capacity==0
        self.buffer = deque(maxlen=maxlen)          # => buffer just 1 transition

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done)) #deque => if full, delete oldest transition and add new one

    def sample(self, batch_size: int):
        if self.no_replay:
            # Ablation: single most-recent transition only
            batch = [self.buffer[-1]]
        else: #sampling number of transition equal to batch size randomly
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch) #transpone list of tuples => from [(s1,a1,r1,...), (s2,a2,r2,...)] to ([s1,s2,...], [a1,a2,...], ...)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32).to(DEVICE),
            torch.tensor(actions,            dtype=torch.long).to(DEVICE),
            torch.tensor(rewards,            dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(next_obs), dtype=torch.float32).to(DEVICE),
            torch.tensor(dones,              dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ──────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    DQN agent (Mnih et al., 2015) with:
      - epsilon-greedy exploration  (configurable schedule)
      - experience replay           (configurable size; 0 = ablation)
      - target network              (periodic hard update)
      - state visitation tracking   (for exploration analysis)
    """
    def __init__(self, obs_shape, n_actions, config: dict):
        self.obs_shape  = obs_shape
        self.n_actions  = n_actions
        self.config     = config

        #2 identical networks
        self.q_net      = QNetwork(obs_shape, n_actions).to(DEVICE) 
        self.target_net = QNetwork(obs_shape, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict()) #initialized with same weights
        self.target_net.eval() #evaluation mode => disabled dropout anf batch norm if exist

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=config["lr"]) #Optimizer just for q_net; target_net never updated by gradient descendent
        self.replay     = ReplayBuffer(config["buffer_size"], obs_shape)
        self.tracker    = StateVisitationTracker(obs_shape)
        self.step_count = 0

    # ── Action selection ───────────────────────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        eps = self.get_epsilon()
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1) #random action
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE) #add batch dimension => from (C,H,W) to (1,C,H,W)
        with torch.no_grad(): #disabled gradient computation
            return self.q_net(obs_t).argmax(dim=1).item() #greedy action

    def get_epsilon(self) -> float: #epsilon linear dacay => computed using linear interpolation to determine at which step epsilon reaches the final value
        """Linear decay: eps_start -> eps_end over eps_decay_steps steps."""
        progress = min(self.step_count / self.config["eps_decay_steps"], 1.0)
        return ( #starting progess=0 => epsilon = eps.start; 
            self.config["eps_end"]
            + (self.config["eps_start"] - self.config["eps_end"]) * (1 - progress)
        )

    # ── Learning step ──────────────────────────────────────────────────────────
    def update(self, batch_size: int) -> float:
        """
        One gradient step of DQN.

        Implements the semi-gradient TD update (Mnih et al., 2015):
            Loss = MSE( Q(s,a),  r + gamma * max_a' Q_target(s',a') * (1-done) )

        The target network is frozen during the update to stabilise training.
        """
        if len(self.replay) < batch_size: #waiting replay has at least number of transition = batch size for updating 
            return 0.0

        #sampling action from replay buffer
        obs, actions, rewards, next_obs, dones = self.replay.sample(batch_size)

        # Q-values for actions actually taken 
        """
        self.q_net(obs) produces all Q-values (batch, n_actions). 
        gather(1, actions.unsqueeze(1)) select Q-value of taken action — implements Q(s, a). 
        squeeze(1) removes extra dimension → (batch,)
        """
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrap targets using target network — no gradient through target network
        """
        Compute bootstrap targets using target network. 
        torch.no_grad() is essential — targets must not have gradients, otherwise the loss would always be zero (minimising the distance between something and itself). 
        .max(dim=1)[0] takes the maximum Q-value for each state — it implements max_a' Q_target(s', a').
        (1 - dones) sets the bootstrap term to zero for terminal states — if the episode has ended, there is no subsequent state.
        """
        with torch.no_grad():
            max_next_q = self.target_net(next_obs).max(dim=1)[0]
            targets    = rewards + self.config["gamma"] * max_next_q * (1 - dones)

        #TD loss — MSE bretween predicted Q-values and target. 
        #Bellman equetion, loss: L = E[(Q(s,a) - (r + γ * max_a' Q_target(s',a'))²]
        loss = F.mse_loss(q_values, targets)

        #Standard PyTorch updating loop: reset the gradients, compute the gradients via backpropagation, clip the gradients to max_norm=10 to prevent exploding gradients, update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Periodic update of target network
        if self.step_count % self.config["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict()) #copy weights of q_net in target_net 

        return loss.item()


# ── Training loop ──────────────────────────────────────────────────────────────
def train_dqn(config: dict, run_name: str, use_wandb: bool = True):
    """
    Full DQN training loop.

    Required config keys:
        game, seed, total_steps, batch_size, buffer_size,
        lr, gamma, eps_start, eps_end, eps_decay_steps,
        target_update_freq, snapshot_steps (list of ints)

    Returns:
        episode_returns : list of floats
        agent.tracker   : StateVisitationTracker
    """

    set_seed(config["seed"])

    if use_wandb:
        wandb.init(project="project_DQNvsPPO", name=run_name, config=config, reinit=True)

    env   = MinAtarEnv(config["game"], seed=config["seed"])
    agent = DQNAgent(env.obs_shape, env.n_actions, config)

    episode_returns = []
    episode_return  = 0.0
    episode_count   = 0
    obs = env.reset()

    for step in range(1, config["total_steps"] + 1):
        agent.step_count = step
        #updating heatmap
        agent.tracker.update(obs)
        
        #selection and execution acion
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        episode_return += reward

        #appending transition to replay buffer
        agent.replay.push(obs, action, reward, next_obs, done)
        obs = next_obs
        
        #updating network (q_net)
        loss = agent.update(config["batch_size"])

        #ended episode
        if done:
            episode_returns.append(episode_return)
            episode_count += 1

            if use_wandb:
                wandb.log({
                    "episode_return": episode_return,
                    "epsilon":        agent.get_epsilon(),
                    "buffer_size":    len(agent.replay),
                    "loss":           loss,
                    "step":           step,
                    "episode":        episode_count,
                })

            #reset return and env
            episode_return = 0.0
            obs = env.reset()

        #save snapshot of heatmap if right step
        if step in config.get("snapshot_steps", []):
            agent.tracker.snapshot(step)

        if step % 10_000 == 0:
            recent = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            print(
                f"Step {step:>7,} | Episodes: {episode_count:>4} "
                f"| Avg return (last 20): {recent:.2f} "
                f"| Eps: {agent.get_epsilon():.3f}"
            )

    if use_wandb:
        wandb.finish()

    return episode_returns, agent.tracker
