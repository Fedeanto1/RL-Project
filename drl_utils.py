"""
DRL Project — Shared Utilities
================================
Helpers used by both DQN and PPO:
  - set_seed
  - MinAtarEnv wrapper
  - StateVisitationTracker
  - Plotting functions
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from minatar import Environment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[utils] Using device: {DEVICE}")


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ── MinAtar wrapper ────────────────────────────────────────────────────────────
class MinAtarEnv:
    """
    Thin wrapper around MinAtar providing a gym-like interface.
    Observation shape: (C, H, W)  float32  — ready for a CNN.
    """
    def __init__(self, game: str, seed: int = 0):
        self.env = Environment(game)
        self.n_actions = self.env.num_actions()
        obs = self.env.state()                          # (H, W, C)
        self.obs_shape = (obs.shape[2], obs.shape[0], obs.shape[1])  # (C, H, W)

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def step(self, action: int):
        reward, done = self.env.act(action)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = self.env.state().astype(np.float32)
        return np.transpose(obs, (2, 0, 1))             # (C, H, W)


# ── State Visitation Tracker ───────────────────────────────────────────────────
class StateVisitationTracker:
    """
    Accumulates a per-cell activation count over all observations seen
    during training.  Used to produce exploration heatmaps.

    obs_shape : (C, H, W)
    """
    def __init__(self, obs_shape):
        _, H, W = obs_shape
        self.heatmap = np.zeros((H, W), dtype=np.float32)
        self.snapshots = []          # list of (step, normalised_heatmap)

    def update(self, obs: np.ndarray):
        """obs: (C, H, W) — collapse channels with max."""
        self.heatmap += obs.max(axis=0)

    def snapshot(self, step: int):
        """Save a normalised copy of the current heatmap."""
        h = self.heatmap.copy()
        if h.max() > 0:
            h /= h.max()
        self.snapshots.append((step, h))

    def plot_snapshots(self, title_prefix: str = ""):
        n = len(self.snapshots)
        if n == 0:
            print("No snapshots recorded yet.")
            return
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (step, h) in zip(axes, self.snapshots):
            im = ax.imshow(h, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"{title_prefix}\nStep {step:,}")
            ax.axis("off")
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        plt.suptitle("State Visitation Heatmap (normalised)", y=1.02)
        plt.tight_layout()
        plt.show()


# ── Plotting utilities ─────────────────────────────────────────────────────────
def plot_returns(returns_dict: dict, title: str = "Training Returns", window: int = 10):
    """
    Smoothed return curves.
    returns_dict : {label: list_of_episode_returns}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, returns in returns_dict.items():
        arr = np.array(returns, dtype=np.float32)
        smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_returns_with_std(
    all_runs: dict,
    title: str = "Training Returns (mean +/- std)",
    window: int = 10,
):
    """
    Mean +/- std across multiple seeds (shaded band).
    all_runs : {label: list_of_lists}  — one inner list per seed
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, runs in all_runs.items():
        min_len = min(len(r) for r in runs)
        arr = np.array([r[:min_len] for r in runs])     # (n_seeds, episodes)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        k = np.ones(window) / window
        mean_s = np.convolve(mean, k, mode="valid")
        std_s  = np.convolve(std,  k, mode="valid")
        x = np.arange(len(mean_s))
        ax.plot(x, mean_s, label=label)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def summarise_results(results_dict: dict, label: str):
    """Print mean +/- std of the last-20-episode return for each config."""
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    for cfg_label, runs in results_dict.items():
        finals = [np.mean(r[-20:]) for r in runs if len(r) >= 20]
        if finals:
            print(f"  {cfg_label:<28} {np.mean(finals):.2f} +/- {np.std(finals):.2f}")


# ── Sample Efficiency ──────────────────────────────────────────────────────────
def area_under_curve(returns: list) -> float:
    """
    Compute the area under the return curve (AUC) normalised by length.
    Higher AUC = better sample efficiency (learns more with fewer episodes).
    Uses the trapezoidal rule via numpy.
    """
    arr = np.array(returns, dtype=np.float32)
    return float(np.trapz(arr) / len(arr))


def sample_efficiency_summary(results_dict: dict, label: str):
    """
    Compare sample efficiency between configurations using AUC.
    Prints mean +/- std of both final return and AUC for each config.
    Use this for the cross-algorithm comparison (Exp 7).
    """
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  {'Config':<28} {'Final Return':>15} {'AUC':>15}")
    print(f"  {'-'*58}")
    for cfg_label, runs in results_dict.items():
        finals = [np.mean(r[-20:]) for r in runs if len(r) >= 20]
        aucs   = [area_under_curve(r) for r in runs if len(r) > 0]
        if finals:
            print(f"  {cfg_label:<28} "
                  f"{np.mean(finals):>6.2f} +/- {np.std(finals):<6.2f}   "
                  f"{np.mean(aucs):>6.2f} +/- {np.std(aucs):<6.2f}")
