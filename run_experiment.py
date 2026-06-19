"""
run_experiment.py — CLI entry point for all experimental conditions.

Usage examples:

  # Run all 15 conditions (4-8 hours)
  python run_experiment.py --all --ollama-url http://192.168.1.5:11434

  # Quick smoke test (50 episodes, no LLM)
  python run_experiment.py --condition ib_beta0_n8 --episodes 50 --no-llm

  # Single Variant A run (fast)
  python run_experiment.py --condition varA_n8_s0_l01 --fast --ollama-url http://192.168.1.5:11434

  # Only baseline conditions
  python run_experiment.py --set baselines

  # Only Variant A conditions
  python run_experiment.py --set variant_a

  # Generate plots after all runs complete
  python run_experiment.py --plots-only
"""
from __future__ import annotations

import sys
# PyTorch 2.x dynamo compatibility patch for certain Python builds
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    def set_int_max_str_digits(maxdigits: int) -> None:
        pass
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits
import argparse
import copy
import json
import os
import sys

from config import (
    ExperimentConfig, EnvConfig, ChannelConfig, NetworkConfig,
    TrainingConfig, IBConfig, VariantAConfig, EvalConfig,
    get_results_dir,
)
from train import train
from evaluate import evaluate
from plot_results import (
    plot_heatmap, plot_training_curves, plot_llm_weights,
    plot_lambda_ablation, plot_loss_decomposition,
)


# ---------------------------------------------------------------------------
# Experiment condition registry
# ---------------------------------------------------------------------------

def _make_config(
    name: str,
    n: int,
    sigma2: float,
    beta: float,
    use_ib: bool,
    use_variant_a: bool,
    lambda_llm: float = 0.0,
    use_fixed_weights: bool = False,   # Ablation: fixed physics weights, no LLM query
    n_episodes: int = 3000,
    seed: int = 42,
    ollama_url: str = "http://localhost:11434",
) -> ExperimentConfig:
    """Build an ExperimentConfig for a named condition."""
    variant_a_cfg = VariantAConfig(
        enabled=use_variant_a,
        lambda_llm=lambda_llm,
        ollama_url=ollama_url,
        # For ablation: set query_interval very large so LLM is never queried
        llm_query_interval=999999 if use_fixed_weights else 50,
    )

    return ExperimentConfig(
        env=EnvConfig(pole_length=0.5, force_magnitude=10.0),
        channel=ChannelConfig(n=n, sigma2_channel=sigma2),
        training=TrainingConfig(n_episodes=n_episodes),
        ib=IBConfig(beta=beta, use_ib=use_ib),
        variant_a=variant_a_cfg,
        seed=seed,
        experiment_name=name,
        results_dir="results",
    )


# Condition definitions: (name, n, sigma2, beta, use_ib, use_variant_a, lambda_llm)
BASELINES = [
    ("non_ib_n8",       8,  0.0, 0.0,        False, False, 0.0),
    ("ib_beta0_n8",     8,  0.0, 0.0,        True,  False, 0.0),
    ("ib_opt_n8_s0",    8,  0.0, 1.9885e-6,  True,  False, 0.0),
    ("ib_beta0_n8_s2",  8,  2.0, 0.0,        True,  False, 0.0),
    ("ib_opt_n8_s2",    8,  2.0, 6.253e-3,   True,  False, 0.0),
    ("ib_opt_n16_s2",  16,  2.0, 3.9134e-9,  True,  False, 0.0),
]

VARIANT_A = [
    ("varA_n8_s0_l001",  8,  0.0, 1.9885e-6, True, True,  0.01),
    ("varA_n8_s0_l01",   8,  0.0, 1.9885e-6, True, True,  0.10),
    ("varA_n8_s0_l05",   8,  0.0, 1.9885e-6, True, True,  0.50),
    ("varA_n8_s2_l001",  8,  2.0, 6.253e-3,  True, True,  0.01),
    ("varA_n8_s2_l01",   8,  2.0, 6.253e-3,  True, True,  0.10),
    ("varA_n16_s2_l01", 16,  2.0, 3.9134e-9, True, True,  0.10),
]

ABLATION = [
    # Fixed physics weights (no LLM query) — isolates LLM adaptation benefit
    ("ablation_fixed_n8_s0",  8,  0.0, 1.9885e-6, True, True, 0.10, True),
    ("ablation_fixed_n8_s2",  8,  2.0, 6.253e-3,  True, True, 0.10, True),
]

ALL_CONDITIONS = BASELINES + VARIANT_A + ABLATION

# Condition index
CONDITION_MAP = {row[0]: row for row in ALL_CONDITIONS}


# ---------------------------------------------------------------------------
# Paper baselines for comparison table
# ---------------------------------------------------------------------------

PAPER_BASELINES = {
    "non_ib_n8":      84.96,
    "ib_beta0_n8":   122.11,
    "ib_opt_n8_s0":  141.56,
    "ib_beta0_n8_s2": 21.88,
    "ib_opt_n8_s2":   90.96,
    "ib_opt_n16_s2": 104.32,
}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _load_overall_avg(name: str) -> str:
    path = os.path.join("results", name, "overall_avg.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return "--"


def print_comparison_table(completed: list) -> None:
    """Print a formatted comparison table for all completed experiments."""
    print("\n" + "=" * 90)
    print(f"{'Method':<28} {'n':>3} {'sigma2':>8} {'beta':>12} {'lambda':>8} {'Avg Reward':>12} {'vs Paper Best':>15}")
    print("-" * 90)

    # Paper baselines first
    paper_rows = [
        ("Non-IB DQN",           8,  0.0, "--",        "--",    84.96, "baseline"),
        ("IB, beta=0",           8,  0.0, "0",         "--",   122.11, "+43.7%"),
        ("IB, beta opt",         8,  0.0, "1.99e-6",   "--",   141.56, "+66.7%"),
        ("IB, beta=0 (s2=2)",    8,  2.0, "0",         "--",    21.88, "baseline"),
        ("IB, beta opt (s2=2)",  8,  2.0, "6.25e-3",   "--",    90.96, "+315.6%"),
        ("IB, beta opt (n16s2)", 16, 2.0, "3.91e-9",   "--",  104.32, "baseline"),
    ]
    print("[Paper results]")
    for row in paper_rows:
        name, n, s2, beta, lam, avg, vs = row
        print(f"  {name:<26} {n:>3} {s2:>8.1f} {beta:>12} {lam:>8} {avg:>12.2f} {vs:>15}")

    print("\n[This run]")
    for name in completed:
        avg_str = _load_overall_avg(name)
        row = CONDITION_MAP.get(name)
        if row is None:
            continue
        _, n, s2, beta, use_ib, use_va, lam = row[:7]
        avg_val = float(avg_str) if avg_str != "--" else None

        # Find relevant paper best to compare against
        paper_best = None
        if name in PAPER_BASELINES:
            paper_best = PAPER_BASELINES[name]
        elif "n8_s0" in name:
            paper_best = 141.56
        elif "n8_s2" in name:
            paper_best = 90.96
        elif "n16_s2" in name:
            paper_best = 104.32

        vs_str = "—"
        if avg_val is not None and paper_best is not None and paper_best > 0:
            pct = (avg_val - paper_best) / paper_best * 100
            vs_str = f"{pct:+.1f}%"

        beta_str = f"{beta:.2e}" if beta > 0 else "0"
        lam_str  = f"{lam:.3f}" if lam > 0 else "--"
        print(
            f"  {name:<26} {n:>3} {s2:>8.1f} {beta_str:>12} {lam_str:>8} "
            f"{avg_str:>12} {vs_str:>15}"
        )

    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RL-over-Noisy-Channels experiments (Variant A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Run selection
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all",       action="store_true", help="Run all 15 conditions.")
    sel.add_argument("--set",       choices=["baselines", "variant_a", "ablation"],
                     help="Run one set of conditions.")
    sel.add_argument("--condition", type=str, help="Run a single named condition.")
    sel.add_argument("--plots-only", action="store_true", help="Only generate plots, no training.")

    # Overrides
    p.add_argument("--episodes",   type=int, default=None, help="Override n_episodes.")
    p.add_argument("--fast",       action="store_true",    help="Set n_episodes=500.")
    p.add_argument("--no-llm",     action="store_true",    help="Disable Variant A LLM (use fixed weights / fallback).")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                   help="Ollama server URL (e.g. http://192.168.1.5:11434).")
    p.add_argument("--no-eval",    action="store_true",
                   help="Skip generalization grid evaluation after training.")
    p.add_argument("--no-plots",   action="store_true", help="Skip plot generation.")

    return p.parse_args()


def run_condition(row: tuple, args: argparse.Namespace) -> str:
    """
    Build config, train, evaluate, return experiment name.
    Row format: (name, n, sigma2, beta, use_ib, use_variant_a, lambda_llm[, use_fixed])
    """
    use_fixed = False
    if len(row) >= 8:
        use_fixed = bool(row[7])

    name, n, sigma2, beta, use_ib, use_variant_a, lambda_llm = row[:7]

    n_episodes = 500 if args.fast else (args.episodes or 3000)

    cfg = _make_config(
        name=name,
        n=n,
        sigma2=sigma2,
        beta=beta,
        use_ib=use_ib,
        use_variant_a=use_variant_a and not args.no_llm,
        lambda_llm=lambda_llm,
        use_fixed_weights=use_fixed or args.no_llm,
        n_episodes=n_episodes,
        seed=args.seed,
        ollama_url=args.ollama_url,
    )

    # If --no-llm: keep Variant A structure but use fallback weights only
    if args.no_llm and use_variant_a:
        cfg.variant_a.enabled = True        # Keep recon head for loss structure
        cfg.variant_a.llm_query_interval = 999999  # Never actually query

    # Train
    train(cfg, ollama_url=args.ollama_url)

    # Evaluate on generalization grid
    if not args.no_eval:
        evaluate(cfg)

    return name


def main() -> None:
    args = parse_args()

    # --- Plots only mode ---
    if args.plots_only:
        print("[run] Plots-only mode.")
        results_dir = "results"
        plots_dir   = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        exp_dirs = {}
        for nm in sorted(os.listdir(results_dir)):
            d = os.path.join(results_dir, nm)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "training_log.jsonl")):
                exp_dirs[nm] = d
        plot_training_curves(exp_dirs, plots_dir=plots_dir)
        for nm, rdir in exp_dirs.items():
            if os.path.exists(os.path.join(rdir, "eval_grid.json")):
                plot_heatmap(rdir, nm, plots_dir=plots_dir)
            if "varA" in nm or "ablation" in nm:
                plot_llm_weights(rdir, nm, plots_dir=plots_dir)
        print_comparison_table(list(exp_dirs.keys()))
        return

    # --- Select conditions to run ---
    if args.all:
        conditions = ALL_CONDITIONS
    elif args.set == "baselines":
        conditions = BASELINES
    elif args.set == "variant_a":
        conditions = VARIANT_A
    elif args.set == "ablation":
        conditions = ABLATION
    elif args.condition:
        if args.condition not in CONDITION_MAP:
            print(f"ERROR: Unknown condition '{args.condition}'. Valid options:")
            for k in CONDITION_MAP:
                print(f"  {k}")
            sys.exit(1)
        conditions = [CONDITION_MAP[args.condition]]
    else:
        print("ERROR: Specify --all, --set, --condition, or --plots-only.")
        sys.exit(1)

    # --- Run ---
    completed = []
    for row in conditions:
        try:
            name = run_condition(row, args)
            completed.append(name)
        except KeyboardInterrupt:
            print("\n[run] Interrupted. Saving results for completed conditions.")
            break
        except Exception as exc:
            print(f"\n[run] ERROR in condition '{row[0]}': {exc}")
            import traceback; traceback.print_exc()
            completed.append(row[0])   # Still include in table with missing avg

    # --- Final comparison table ---
    print_comparison_table(completed)

    # --- Generate plots ---
    if not args.no_plots and completed:
        print("[run] Generating plots...")
        results_dir = "results"
        plots_dir   = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        exp_dirs = {nm: os.path.join(results_dir, nm) for nm in completed}
        plot_training_curves(exp_dirs, plots_dir=plots_dir)

        for nm in completed:
            rdir = os.path.join(results_dir, nm)
            if os.path.exists(os.path.join(rdir, "eval_grid.json")):
                row = CONDITION_MAP.get(nm)
                label = f"n={row[1]}, sigma2={row[2]}, beta={row[3]:.2e}" if row else ""
                plot_heatmap(rdir, nm, label=label, plots_dir=plots_dir)
            if "varA" in nm or "ablation" in nm:
                plot_llm_weights(rdir, nm, plots_dir=plots_dir)
                cfg_path = os.path.join(rdir, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        cfg_j = json.load(f)
                    beta = cfg_j.get("ib", {}).get("beta", 0.0)
                    lam  = cfg_j.get("variant_a", {}).get("lambda_llm", 0.0)
                    plot_loss_decomposition(rdir, nm, beta=beta, lambda_llm=lam, plots_dir=plots_dir)

        # Lambda ablation plots (group by (n, sigma2))
        _make_lambda_ablation_plots(completed, plots_dir)

        print(f"[run] Plots saved to {plots_dir}/")


def _make_lambda_ablation_plots(completed: list, plots_dir: str) -> None:
    """Generate lambda ablation plots for available Variant A conditions."""
    # Group: (n, sigma2) -> {lambda: overall_avg}
    groups: dict = {}
    baseline_avgs: dict = {}

    for name in completed:
        row = CONDITION_MAP.get(name)
        if row is None:
            continue
        _, n, sigma2, beta, use_ib, use_va, lam = row[:7]

        avg_path = os.path.join("results", name, "overall_avg.txt")
        if not os.path.exists(avg_path):
            continue
        with open(avg_path) as f:
            avg = float(f.read().strip())

        key = (n, sigma2)
        if use_va and lam > 0:
            groups.setdefault(key, {})[lam] = avg
        elif not use_va and use_ib:
            baseline_avgs[key] = avg

    for key, lam_avgs in groups.items():
        if len(lam_avgs) < 2:
            continue
        n, sigma2 = key
        baseline = baseline_avgs.get(key, 0.0)
        plot_lambda_ablation(lam_avgs, baseline_avg=baseline, sigma2=sigma2, n=n, plots_dir=plots_dir)


if __name__ == "__main__":
    main()
