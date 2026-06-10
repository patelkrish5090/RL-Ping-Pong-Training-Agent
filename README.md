# Variant A: LLM-Guided Semantic Encoder Reward

Extension of **"Reinforcement Learning over Noisy Channels: An Information Bottleneck Approach"** (Kam, Macker, Sun — MILCOM 2024).

---

## Overview

This project trains a JSCC-DQN agent for CartPole over an AWGN channel, augmented with a physics-aware auxiliary loss whose per-feature importance weights are dynamically provided by a local **Qwen3-Coder** LLM (via Ollama) every 50 training episodes.

**Combined objective:**
```
L_total = L_DQN(ψ) + β · I_θ(ẑ; x) + λ · L_llm(θ, g)
```

Where `L_llm = Σᵢ wᵢ · E[(xᵢ - ĝᵢ(ẑ))²]` and `w` are LLM-assigned importance weights for each CartPole state feature.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with qwen3-coder:latest on your server
# (no API key required)
```

---

## Quick Start

```bash
# Smoke test (50 episodes, no LLM, just verify pipeline)
python run_experiment.py --condition ib_beta0_n8 --episodes 50 --no-llm

# Single Variant A run (fast, 500 episodes)
python run_experiment.py --condition varA_n8_s0_l01 --fast \
    --ollama-url http://192.168.x.x:11434

# Run all paper baselines (reproduce Table I)
python run_experiment.py --set baselines --ollama-url http://192.168.x.x:11434

# Run all Variant A conditions
python run_experiment.py --set variant_a --ollama-url http://192.168.x.x:11434

# Run all 15 conditions (4-8 hours)
python run_experiment.py --all --ollama-url http://192.168.x.x:11434

# Evaluate a saved checkpoint on the generalization grid
python evaluate.py --checkpoint results/varA_n8_s0_l01/model.pt \
    --name varA_n8_s0_l01 --n 8 --sigma2 0.0

# Generate all plots from existing results
python run_experiment.py --plots-only
python plot_results.py --results_dir results/
```

---

## Repository Structure

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters as dataclasses |
| `channel.py` | Non-trainable AWGN channel (reparameterized) |
| `encoder.py` | JSCC encoder with Tanh power constraint |
| `controller.py` | Q-network controller (receiver) |
| `reconstruction_head.py` | Auxiliary feature reconstruction head (training only) |
| `ib_loss.py` | Nonlinear IB mutual information upper bound |
| `llm_advisor.py` | Ollama LLM weight advisor with caching & fallback |
| `dqn_agent.py` | End-to-end DQN with combined loss |
| `train.py` | Training loop |
| `evaluate.py` | Generalization evaluation grid |
| `plot_results.py` | Publication-quality figures |
| `run_experiment.py` | CLI entry point, all 15 conditions |

---

## Experiment Conditions

### Set 1: Paper Baselines
| Name | n | σ² | β | Expected avg |
|---|---|---|---|---|
| `non_ib_n8` | 8 | 0.0 | 0 | ~84.96 |
| `ib_beta0_n8` | 8 | 0.0 | 0 | ~122.11 |
| `ib_opt_n8_s0` | 8 | 0.0 | 1.99e-6 | ~141.56 |
| `ib_beta0_n8_s2` | 8 | 2.0 | 0 | ~21.88 |
| `ib_opt_n8_s2` | 8 | 2.0 | 6.25e-3 | ~90.96 |
| `ib_opt_n16_s2` | 16 | 2.0 | 3.91e-9 | ~104.32 |

### Set 2: Variant A (LLM-guided)
| Name | n | σ² | β | λ |
|---|---|---|---|---|
| `varA_n8_s0_l001` | 8 | 0 | 1.99e-6 | 0.01 |
| `varA_n8_s0_l01` | 8 | 0 | 1.99e-6 | 0.10 |
| `varA_n8_s0_l05` | 8 | 0 | 1.99e-6 | 0.50 |
| `varA_n8_s2_l001` | 8 | 2 | 6.25e-3 | 0.01 |
| `varA_n8_s2_l01` | 8 | 2 | 6.25e-3 | 0.10 |
| `varA_n16_s2_l01` | 16 | 2 | 3.91e-9 | 0.10 |

### Set 3: Ablation (fixed physics weights, no LLM)
| Name | n | σ² | Note |
|---|---|---|---|
| `ablation_fixed_n8_s0` | 8 | 0 | Isolates LLM adaptation benefit |
| `ablation_fixed_n8_s2` | 8 | 2 | Fixed: [0.08, 0.20, 0.42, 0.30] |

---

## Key Design Choices

1. **Shared noise sample**: One reparameterized channel draw feeds both `L_DQN` and `L_llm` in each training step — consistent channel realization.
2. **LLM weights as constants**: `w` is a numpy array detached from autograd — only the reconstruction error is differentiated.
3. **Lazy Ollama client**: Connection attempted only on first LLM query — won't fail on `--no-llm` runs.
4. **Fallback weights**: If Ollama is unreachable, physics-based defaults are used: `[0.08, 0.20, 0.42, 0.30]` at high noise, `[0.15, 0.20, 0.35, 0.30]` at low noise.
5. **Reconstruction head discarded at eval**: `model.eval()` + `torch.no_grad()` — no `L_llm` computation during evaluation.

---

## Output Files (per experiment in `results/{name}/`)

| File | Content |
|------|---------|
| `config.json` | Full experiment config |
| `training_log.jsonl` | Per-episode: reward, ε, L_DQN, L_IB, L_llm, LLM weights |
| `llm_log.jsonl` | Per-query: episode, weights, reasoning, source, elapsed |
| `model.pt` | Final checkpoint (encoder + controller + recon_head) |
| `eval_grid.npy` | 2D reward grid (n_forces × n_lengths) |
| `eval_grid.json` | Same as JSON |
| `overall_avg.txt` | Scalar mean reward over generalization grid |

---

## Paper Citation

```bibtex
@inproceedings{kam2024rl_noisy,
  title={Reinforcement Learning over Noisy Channels: An Information Bottleneck Approach},
  author={Kam, Clement and Macker, Joseph P. and Sun, Yin},
  booktitle={MILCOM 2024 - 2024 IEEE Military Communications Conference},
  year={2024},
  doi={10.1109/MILCOM61039.2024.10773737}
}
```
