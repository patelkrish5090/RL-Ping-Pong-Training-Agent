"""
plot_results.py — Generate all publication-quality figures.

Figures produced:
  Fig 1: Heatmap — avg reward vs pole_length × force_magnitude (per experiment)
  Fig 2: Training curves — reward vs episode, 100-ep moving avg
  Fig 3: LLM weight evolution — w₀..w₃ over episodes (Variant A experiments)
  Fig 4: Lambda ablation — overall_avg vs λ for fixed (n, σ², β)
  Fig 5: Loss decomposition — L_DQN, β·L_IB, λ·L_llm over episodes

All saved to results/plots/.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
})

FEATURE_NAMES  = ["x_cart", "ẋ_cart", "θ_pole", "θ̇_pole"]
FEATURE_COLORS = ["#4C9BE8", "#E8914C", "#4CE87A", "#E84C6E"]

# Training point in the evaluation grid (matching paper)
TRAIN_POLE_LEN = 0.5
TRAIN_FORCE    = 10.0


# ---------------------------------------------------------------------------
# Helper: load JSONL log
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _moving_avg(data: List[float], window: int) -> np.ndarray:
    arr = np.array(data, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Fig 1: Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    results_dir: str,
    experiment_name: str,
    label: str = "",
    plots_dir: str = "results/plots",
) -> Optional[str]:
    """
    Plot the eval_grid heatmap for one experiment.

    Parameters
    ----------
    results_dir : str
        Path to the experiment results directory.
    experiment_name : str
    label : str
        Extra label for the figure title.
    plots_dir : str
        Where to save the PNG.

    Returns
    -------
    save_path : str or None
    """
    grid_path = os.path.join(results_dir, "eval_grid.json")
    if not os.path.exists(grid_path):
        print(f"[plot] No eval_grid.json found for {experiment_name}, skipping heatmap.")
        return None

    with open(grid_path) as f:
        data = json.load(f)

    grid        = np.array(data["grid"])
    pole_lens   = data["pole_lengths"]
    force_mags  = data["force_magnitudes"]
    overall_avg = data["overall_avg"]

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        vmin=0, vmax=200,
        cmap="RdYlGn",
        extent=[
            pole_lens[0]  - 0.05, pole_lens[-1]  + 0.05,
            force_mags[0] - 1.0,  force_mags[-1] + 1.0,
        ],
    )
    plt.colorbar(im, ax=ax, label="Mean episode reward (max 200)")

    # Mark training point with a red square
    if TRAIN_POLE_LEN in pole_lens and TRAIN_FORCE in force_mags:
        rect = mpatches.FancyBboxPatch(
            (TRAIN_POLE_LEN - 0.07, TRAIN_FORCE - 1.5),
            0.14, 3.0,
            boxstyle="square,pad=0",
            linewidth=2.5,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            TRAIN_POLE_LEN, TRAIN_FORCE - 3.0,
            "Train",
            color="red", ha="center", va="top", fontsize=8,
        )

    ax.set_xlabel("Pole length (m)", fontsize=11)
    ax.set_ylabel("Force magnitude (N)", fontsize=11)
    ax.set_title(
        f"{experiment_name}  {label}\n"
        f"Overall avg reward = {overall_avg:.2f}",
        fontsize=11,
    )

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"heatmap_{experiment_name}.png")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved heatmap → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Fig 2: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    experiments: Dict[str, str],
    plots_dir: str = "results/plots",
    window: int = 100,
) -> str:
    """
    Plot episode reward (smoothed) for multiple experiments on one axis.

    Parameters
    ----------
    experiments : dict
        {label: results_dir_path}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("tab10")
    for i, (label, rdir) in enumerate(experiments.items()):
        log_path = os.path.join(rdir, "training_log.jsonl")
        if not os.path.exists(log_path):
            print(f"[plot] No training_log.jsonl for {label}, skipping.")
            continue
        entries = _load_jsonl(log_path)
        rewards = [e["reward"] for e in entries]
        episodes = list(range(len(rewards)))

        color = cmap(i % 10)
        # Raw reward (faint)
        ax.plot(episodes, rewards, alpha=0.15, color=color)
        # Smoothed
        if len(rewards) >= window:
            smoothed = _moving_avg(rewards, window)
            ax.plot(
                range(window - 1, len(rewards)),
                smoothed,
                color=color,
                label=f"{label} ({window}-ep avg)",
                linewidth=2,
            )

    ax.axhline(200, color="grey", linestyle="--", linewidth=1, label="Max reward")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode reward", fontsize=11)
    ax.set_title("Training curves comparison", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(bottom=0)

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "training_curves.png")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved training curves → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Fig 3: LLM weight evolution
# ---------------------------------------------------------------------------

def plot_llm_weights(
    results_dir: str,
    experiment_name: str,
    plots_dir: str = "results/plots",
    window: int = 10,
) -> Optional[str]:
    """
    Plot the LLM-assigned importance weights over training episodes.
    """
    llm_log_path = os.path.join(results_dir, "llm_log.jsonl")
    if not os.path.exists(llm_log_path):
        print(f"[plot] No llm_log.jsonl for {experiment_name}, skipping weight plot.")
        return None

    entries = _load_jsonl(llm_log_path)
    if not entries:
        return None

    episodes = [e["episode"] for e in entries]
    weights  = np.array([e["weights"] for e in entries])  # (T, 4)

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (name, color) in enumerate(zip(FEATURE_NAMES, FEATURE_COLORS)):
        ax.step(episodes, weights[:, i], where="post", color=color, label=name, linewidth=2)

    ax.axhline(0.25, color="grey", linestyle="--", linewidth=1, alpha=0.5, label="Uniform (0.25)")
    ax.set_xlabel("Training episode", fontsize=11)
    ax.set_ylabel("Importance weight", fontsize=11)
    ax.set_title(f"LLM weight evolution — {experiment_name}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.70)

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"llm_weights_{experiment_name}.png")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved LLM weight evolution → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Fig 4: Lambda ablation
# ---------------------------------------------------------------------------

def plot_lambda_ablation(
    lambda_results: Dict[float, float],
    baseline_avg: float,
    sigma2: float,
    n: int,
    plots_dir: str = "results/plots",
) -> str:
    """
    Plot overall_avg vs λ (lambda_llm).

    Parameters
    ----------
    lambda_results : dict  {lambda_val: overall_avg_reward}
    baseline_avg : float   Overall avg of the IB baseline (λ=0)
    sigma2 : float
    n : int
    """
    lambdas = sorted(lambda_results.keys())
    avgs    = [lambda_results[l] for l in lambdas]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, avgs, "o-", color="#4C9BE8", linewidth=2, markersize=8, label="Variant A")
    ax.axhline(baseline_avg, color="#E8914C", linestyle="--", linewidth=2, label=f"IB baseline (λ=0): {baseline_avg:.1f}")

    ax.set_xlabel("λ (LLM loss weight)", fontsize=11)
    ax.set_ylabel("Overall avg reward (generalization grid)", fontsize=11)
    ax.set_title(f"λ ablation: n={n}, σ²={sigma2}", fontsize=12)
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"lambda_ablation_n{n}_s{sigma2}.png")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved lambda ablation → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Fig 5: Loss decomposition
# ---------------------------------------------------------------------------

def plot_loss_decomposition(
    results_dir: str,
    experiment_name: str,
    beta: float,
    lambda_llm: float,
    plots_dir: str = "results/plots",
    window: int = 50,
) -> Optional[str]:
    """
    Plot the three loss components (DQN, β·IB, λ·LLM) over training.
    """
    log_path = os.path.join(results_dir, "training_log.jsonl")
    if not os.path.exists(log_path):
        return None

    entries = _load_jsonl(log_path)
    # Filter to entries where losses are present (after replay buffer warms up)
    entries = [e for e in entries if e.get("loss_dqn") is not None]
    if not entries:
        return None

    episodes  = [e["episode"]   for e in entries]
    l_dqn     = [e["loss_dqn"]  for e in entries]
    l_ib      = [beta * e["loss_ib"]  for e in entries]
    l_llm     = [lambda_llm * e["loss_llm"] for e in entries]

    def smooth(lst):
        if len(lst) >= window:
            return list(range(window - 1, len(lst))), _moving_avg(lst, window).tolist()
        return list(range(len(lst))), lst

    fig, ax = plt.subplots(figsize=(10, 4))

    for values, label, color in [
        (l_dqn, "L_DQN",     "#4C9BE8"),
        (l_ib,  "β·L_IB",    "#E8914C"),
        (l_llm, "λ·L_LLM",  "#4CE87A"),
    ]:
        x_raw = episodes
        ax.plot(x_raw, values, alpha=0.15, color=color)
        x_s, y_s = smooth(values)
        ax.plot(
            [episodes[i] for i in x_s], y_s,
            color=color, label=label, linewidth=2,
        )

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Loss (smoothed)", fontsize=11)
    ax.set_title(f"Loss decomposition — {experiment_name}", fontsize=12)
    ax.legend(fontsize=9)

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"loss_decomp_{experiment_name}.png")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved loss decomposition → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI: generate all plots for all available experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate all result plots.")
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Top-level results directory containing per-experiment subdirs.",
    )
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Discover experiment directories
    exp_dirs = {}
    if os.path.isdir(args.results_dir):
        for name in sorted(os.listdir(args.results_dir)):
            d = os.path.join(args.results_dir, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "training_log.jsonl")):
                exp_dirs[name] = d

    print(f"[plot] Found {len(exp_dirs)} experiments: {list(exp_dirs.keys())}")

    # Fig 1: Heatmap for each experiment
    for name, rdir in exp_dirs.items():
        if os.path.exists(os.path.join(rdir, "eval_grid.json")):
            plot_heatmap(rdir, name, plots_dir=plots_dir)

    # Fig 2: Training curves (all experiments together)
    if exp_dirs:
        plot_training_curves(exp_dirs, plots_dir=plots_dir)

    # Fig 3: LLM weight evolution for Variant A experiments
    for name, rdir in exp_dirs.items():
        if "varA" in name or "variant" in name.lower():
            plot_llm_weights(rdir, name, plots_dir=plots_dir)

    # Fig 5: Loss decomposition for Variant A experiments
    for name, rdir in exp_dirs.items():
        cfg_path = os.path.join(rdir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            beta = cfg.get("ib", {}).get("beta", 0.0)
            lam  = cfg.get("variant_a", {}).get("lambda_llm", 0.0)
            plot_loss_decomposition(rdir, name, beta=beta, lambda_llm=lam, plots_dir=plots_dir)

    print(f"\n[plot] All figures saved to {plots_dir}/")
