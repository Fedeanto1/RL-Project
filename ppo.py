"""
DRL Project — PPO
==================
Based on CleanRL's PPO implementation:
  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

Original additions:
  - Configurable hyperparameter dictionary replacing CleanRL's hardcoded values
  - StateVisitationTracker integration for exploration heatmaps
  - No-clip ablation (clip_coef=1.0 = REINFORCE-like ablation)
  - No-entropy ablation (ent_coef=0.0 = no exploration bonus)
  - W&B logging of custom metrics (policy entropy, value loss, policy loss)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from drl_utils import DEVICE, set_seed, MinAtarEnv, StateVisitationTracker


# ── Actor-Critic Network ───────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared CNN backbone with separate actor and critic heads.

    Architecture: Conv(C->16, 3x3) -> ReLU -> Flatten -> Linear(16HW->128)
                  -> ReLU -> [actor: Linear(128->n_actions)]
                          -> [critic: Linear(128->1)]

    Input : (batch, C, H, W)
    Output: action logits (batch, n_actions),  state value (batch, 1)
    """
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        C, H, W = obs_shape
        self.shared = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * H * W, 128),
            nn.ReLU(),
        ) 
        #same architecture of dqn but at the end two different heads, but sharing same feature extracted from backbone:
        #Actor — produce logits of policy: probability for each action
        #Critic — produce V(s): how much is good to stay in the state

        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        feat   = self.shared(x) #features generated once by backbone and used to produce logits and values
        logits = self.actor(feat) #(batch, n_actions) - logits no normalised
        value  = self.critic(feat) #(batch, 1) - value state
        return logits, value

    def get_action_and_value(self, x, action=None):
        """
        This method is used in two different contexts:
        during rollout (action=None) — samples an action from the distribution and returns log_prob and entropy;
        during update (action=act_buf[mb]) — recalculates log_prob and entropy for actions already taken, using the current policy
        """
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits) #creates a categorical distribution from the logits — applies the softmax function internally
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value #dist.log_prob(action) calculate log of the probability of the action
                                                                    #dist.entropy() calculate the entropy of the distribution 


# ── Training loop ──────────────────────────────────────────────────────────────
def train_ppo(config: dict, run_name: str, use_wandb: bool = True):
    """
    PPO training loop (on-policy)

    Key components:
      1. Rollout collection   — interact with env for n_steps steps
      2. GAE advantage        — A_t = sum (gamma*lambda)^l * delta_{t+l}
      3. PPO-Clip update      — L = min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
      4. Entropy bonus        — encourages exploration; ent_coef=0 = ablation
      5. Value function loss  — MSE(V(s), discounted returns)

    clip_coef=1.0 disables clipping entirely (REINFORCE-like ablation).

    Required config keys:
        game, seed, total_steps, n_steps, n_epochs, batch_size,
        lr, gamma, gae_lambda, clip_coef, ent_coef, vf_coef,
        max_grad_norm, snapshot_steps (list of ints)

    Returns:
        episode_returns : list of floats
        tracker         : StateVisitationTracker
    """
    set_seed(config["seed"])

    if use_wandb:
        wandb.init(project="project_DQNvsPPO", name=run_name, config=config, reinit=True)

    env     = MinAtarEnv(config["game"], seed=config["seed"])
    model   = ActorCritic(env.obs_shape, env.n_actions).to(DEVICE)
    optim_  = optim.Adam(model.parameters(), lr=config["lr"], eps=1e-5)
    tracker = StateVisitationTracker(env.obs_shape)

    N = config["n_steps"]
    C, H, W = env.obs_shape

# Pre-allocate rollout buffers on GPU - more efficient than creating tensor at each step  
    obs_buf  = torch.zeros((N, C, H, W), device=DEVICE)
    act_buf  = torch.zeros(N,            device=DEVICE, dtype=torch.long)
    logp_buf = torch.zeros(N,            device=DEVICE)
    rew_buf  = torch.zeros(N,            device=DEVICE)
    done_buf = torch.zeros(N,            device=DEVICE)
    val_buf  = torch.zeros(N,            device=DEVICE)

    episode_returns = []
    episode_return  = 0.0
    episode_count   = 0
    global_step     = 0

    obs = torch.tensor(env.reset(), dtype=torch.float32).to(DEVICE)

    while global_step < config["total_steps"]:

        # ── Collect rollout ────────────────────────────────────────────────────
        for t in range(N):
            global_step += 1
            obs_buf[t] = obs #save obs
            tracker.update(obs.cpu().numpy())

            with torch.no_grad(): #obtain action and value from net with no gradient
                action, logp, _, value = model.get_action_and_value(obs.unsqueeze(0))

            #save action, log_prob and value in buffers
            act_buf[t]  = action
            logp_buf[t] = logp
            val_buf[t]  = value.squeeze()

            #execution action and save reward, next_obs and if done
            next_obs_np, reward, done, _ = env.step(action.item())
            rew_buf[t]  = reward
            done_buf[t] = float(done)
            episode_return += reward

            if done: #rollout continues even if episode ends
                episode_returns.append(episode_return)
                episode_count  += 1
                episode_return  = 0.0
                next_obs_np     = env.reset()

            obs = torch.tensor(next_obs_np, dtype=torch.float32).to(DEVICE)

            if global_step in config.get("snapshot_steps", []):
                tracker.snapshot(global_step)

        # ── GAE Advantage Estimation ───────────────────────────────────────────
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        # A_t     = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
        with torch.no_grad(): #compute value last state of rollout - used as bootstrap for GAE of last step of rollout
            _, next_value = model.forward(obs.unsqueeze(0))
            next_value = next_value.squeeze()

        advantages = torch.zeros(N, device=DEVICE)
        last_gae   = 0.0
        for t in reversed(range(N)): #compiutation in reversed order, from N-1 to 0 
            nv       = next_value if t == N - 1 else val_buf[t + 1]
            nd       = done_buf[t]
            #TD error: difference between reward and predected value
            delta    = rew_buf[t] + config["gamma"] * nv * (1 - nd) - val_buf[t]
            #GAE: accumula i TD error pesati esponenzialmente. 
            last_gae = delta + config["gamma"] * config["gae_lambda"] * (1 - nd) * last_gae #(1-done_t) = 0 when episode ends -> last_gae = delta.  last_gae tiene traccia del vantaggio accumulato dal passo successivo.
            advantages[t] = last_gae
        returns = advantages + val_buf #returns are target for value function; used for supervised learning on critic

        # ── PPO Update ─────────────────────────────────────────────────────────
        indices = np.arange(N) #array numpy of N elements
        for _ in range(config["n_epochs"]): #reuse indeces (different order) for some epochs -> more sample efficient than REINFORCE but less than DQN that usese replay buffer
            np.random.shuffle(indices) #avoiding mini-batch are always the same
            for start in range(0, N, config["batch_size"]):
                mb = indices[start : start + config["batch_size"]] #array of random indexes for mini batch

                #compute new values with current policy -> different from rollout because already done some updates
                _, new_logp, entropy, new_value = model.get_action_and_value(
                    obs_buf[mb], act_buf[mb]
                )
                new_value = new_value.squeeze()

                # Policy ratio pi_new / pi_old  (in log space for numerical stability) -> measure how much policy has changed for that action
                log_ratio = new_logp - logp_buf[mb] #using log -> log(π_new/π_old) = log(π_new) - log(π_old)
                ratio     = log_ratio.exp()

                # Normalise advantages within mini-batch -> reducing the variance of the updates
                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8) #1e-8 to avoid division by zero

                # PPO-Clip objective (Schulman et al., 2017, eq. 7)
                # L_CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
                clip  = config["clip_coef"]
                #used '-'  beacuse PyTorch minimize loss function
                pg1   = -mb_adv * ratio #no clipping
                pg2   = -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip) #implementing clipping -> ratio between 1-ε and 1+ε
                #used max due to '-' (in paper max due to '+')
                pg_loss = torch.max(pg1, pg2).mean() #take worst case between clipped and not clipped -> pessimistic bound of PPO (more conservative update)-> smaller update

                vf_loss      = F.mse_loss(new_value, returns[mb]) #MSE between predicted value and estimated return (by GAE)
                entropy_loss = -entropy.mean()  #prevents the policy from becoming too deterministic, maintaining exploration. negative because we want to maximise entropy

                #total loss is weighted sum: policy gradient with clipping + value function loss + entrpy bonus
                loss = (
                    pg_loss #update policy to do best actions
                    + config["vf_coef"]  * vf_loss #teaches the critic to estimate the value of the states correctly. It's supervised learning on critic: learns to predict real returns.
                    + config["ent_coef"] * entropy_loss  #penalises policy if becomes too deterministic. Il gradient descent spinge la policy verso distribuzioni più uniformi — più esplorazione. ent_coef=0 disables this term (ablation in Exp 3)
                )
                """
                The three components influence one another in a cycle:
                A better critic → better advantage estimates → a more accurate policy gradient
                More exploration → more diverse experiences → a critic that sees more states → a better estimate
                A better policy → better rollouts → higher returns → a critic that is easier to train
                """

                #standard PyTorch updating
                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optim_.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if use_wandb and episode_returns:
            with torch.no_grad():
                #mean entropy of policy over 64 obs of rollout just completed -> monitoring how much policy is exploring during training
                sample_obs = obs_buf[: min(64, N)]
                logits, _  = model(sample_obs)
                mean_entropy = (
                    torch.distributions.Categorical(logits=logits).entropy().mean().item()
                )
            wandb.log({
                "episode_return": np.mean(episode_returns[-20:]),
                "policy_entropy": mean_entropy, #added to CleanRL
                "value_loss":     vf_loss.item(),
                "policy_loss":    pg_loss.item(),
                "total_loss":     loss.item(),
                "step":           global_step,
                "episode":        episode_count,
            })

        if global_step % 10_000 == 0:
            recent = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            print(
                f"Step {global_step:>7,} | Episodes: {episode_count:>4} "
                f"| Avg return (last 20): {recent:.2f}"
            )

    if use_wandb:
        wandb.finish()

    return episode_returns, tracker
