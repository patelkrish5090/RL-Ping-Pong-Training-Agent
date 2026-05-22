"""
LLM Interface for Reward Function Generation — Anti-Jamming Channel Selection
=============================================================================
Interfaces with a local LLM (Ollama) to iteratively generate and refine
reward functions for tactical wireless anti-jamming channel selection.

Key improvements over v1:
  - Tracks reward version history and injects "previous reward FAILED" context
  - Raises shaping magnitude cap to ±0.3 for bigger behavioral impact
  - Separates the three agent failure modes (jam avoidance, switching, queue) and
    tells the LLM *which specific failure mode* to target each iteration
  - Includes per-metric deltas so the LLM sees whether things are getting better or worse
  - Forces the LLM to justify every magnitude choice against a calibration table
"""
import json
import re
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import LLM_CONFIG, REWARD_FILE, REWARD_HISTORY_DIR


# ---------------------------------------------------------------------------
# Reward history tracker — tells LLM which past rewards failed and why
# ---------------------------------------------------------------------------
class RewardHistory:
    """Tracks performance delta across iterations so the LLM gets 'what failed' context."""

    def __init__(self):
        self.entries: List[dict] = []  # {iter, avg_pdr, jammed_rate, switch_rate, code_snippet}

    def add(self, iteration: int, avg_pdr: float, jammed_rate: float,
            switch_rate: float, code_snippet: str):
        self.entries.append({
            "iter": iteration,
            "avg_pdr": avg_pdr,
            "jammed_rate": jammed_rate,
            "switch_rate": switch_rate,
            "code": code_snippet[:600],  # keep first 600 chars for context
        })

    def get_trend_summary(self) -> str:
        """Return a concise summary of what has improved / stagnated across iterations."""
        if len(self.entries) < 2:
            return "No previous iteration data."

        lines = ["### Reward Iteration History (most recent last)"]
        for e in self.entries[-5:]:   # Show last 5 iterations max
            lines.append(
                f"  Iter {e['iter']:>2}: PDR={e['avg_pdr']:.3f}  "
                f"Jammed={e['jammed_rate']:.3f}  Switch={e['switch_rate']:.3f}/step"
            )

        # Delta from earliest to latest in this window
        first = self.entries[-min(5, len(self.entries))]
        last  = self.entries[-1]
        dpdr  = last["avg_pdr"] - first["avg_pdr"]
        djam  = last["jammed_rate"] - first["jammed_rate"]
        dswi  = last["switch_rate"] - first["switch_rate"]

        lines.append(f"\n  Delta (last 5 iters): PDR {dpdr:+.3f}  "
                     f"Jammed {djam:+.3f}  Switch {dswi:+.3f}")

        if abs(dpdr) < 0.02 and abs(djam) < 0.02:
            lines.append("\n  *** PLATEAU DETECTED: PDR has NOT improved over last 5 iterations. "
                         "The previous reward function strategy is NOT working. "
                         "You MUST try a fundamentally different shaping approach. ***")

        return "\n".join(lines)

    def get_failed_strategies(self) -> str:
        """Summarise what the last 3 reward functions tried and why they failed."""
        if not self.entries:
            return "No previous reward functions attempted."
        recent = self.entries[-3:]
        lines = ["### What Previous Reward Functions Tried (and Failed to Fix)"]
        for e in recent:
            lines.append(f"\n  --- Iteration {e['iter']} (PDR={e['avg_pdr']:.3f}) ---")
            lines.append(f"  Code excerpt: ...{e['code'][:300]}...")
        return "\n".join(lines)


class LLMRewardGenerator:
    """
    Interfaces with local LLM (Ollama) to generate and refine reward functions
    for the anti-jamming channel selection task.
    """

    def __init__(self, model: str = None, host: str = None):
        self.model = model or LLM_CONFIG["model"]
        self.host  = host  or LLM_CONFIG["host"]
        self.temperature = LLM_CONFIG.get("temperature", 0.4)   # Lower = more precise code
        self.num_ctx     = LLM_CONFIG.get("num_ctx", 8192)
        self.iteration   = 0
        self.history     = RewardHistory()
        self._last_pdr   = 0.0
        self._last_jammed = 0.0
        self._last_switch = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reward(self, training_summary: str, current_reward_code: str = None,
                        metrics: dict = None) -> Tuple[str, str]:
        """
        Generate or refine a reward function based on training performance.

        Args:
            training_summary:    Analysis of recent training episodes (text)
            current_reward_code: Current reward function code (if refining)
            metrics:             Dict with keys avg_pdr, avg_jammed_rate, avg_switch_rate

        Returns:
            Tuple of (reward_function_code, reasoning)
        """
        self.iteration += 1

        # Record history before asking LLM
        if metrics and current_reward_code:
            self.history.add(
                iteration=self.iteration - 1,
                avg_pdr=metrics.get("avg_pdr", 0.0),
                jammed_rate=metrics.get("avg_jammed_rate", 0.0),
                switch_rate=metrics.get("avg_switch_rate", 0.0),
                code_snippet=current_reward_code,
            )
            self._last_pdr    = metrics.get("avg_pdr", 0.0)
            self._last_jammed = metrics.get("avg_jammed_rate", 0.0)
            self._last_switch = metrics.get("avg_switch_rate", 0.0)

        prompt = self._build_prompt(training_summary, current_reward_code, metrics)
        response = self._call_ollama(prompt)
        code, reasoning = self._parse_response(response)

        self._save_to_history(code, reasoning, training_summary)
        return code, reasoning

    # ------------------------------------------------------------------
    # Ollama HTTP call
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx":     self.num_ctx,
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out after 600 seconds")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _diagnose_primary_failure(self, metrics: dict = None) -> str:
        """Identify which failure mode is most urgent and return a targeted directive."""
        if not metrics:
            return "Focus on all three failure modes equally."

        pdr    = metrics.get("avg_pdr", 0.0)
        jammed = metrics.get("avg_jammed_rate", 0.0)
        switch = metrics.get("avg_switch_rate", 0.0)
        queue  = metrics.get("avg_queue", 0.0)

        issues = []

        if jammed > 0.20:
            issues.append(
                f"[PRIORITY 1 — JAMMER AVOIDANCE FAILING] Jammed TX rate = {jammed:.3f} "
                f"(target < 0.10). The agent is STILL selecting jammed channels far too often. "
                f"The current jammed-channel penalty is clearly NOT large enough. "
                f"You MUST increase the penalty for selecting recently-jammed channels significantly."
            )

        if switch > 0.40:
            issues.append(
                f"[PRIORITY 2 — EXCESSIVE SWITCHING] Switch rate = {switch:.3f}/step "
                f"(target < 0.20). The agent is hopping channels unnecessarily. "
                f"Previous switching penalties have FAILED to reduce this. "
                f"You MUST increase the switching penalty magnitude and/or add a "
                f"'stay-on-good-channel' reward that is large enough to outweigh the urge to switch."
            )

        if queue > 12:
            issues.append(
                f"[PRIORITY 3 — QUEUE BUILDUP] Avg queue = {queue:.1f} pkts "
                f"(target < 6). The agent is not delivering packets fast enough. "
                f"Add or strengthen a penalty proportional to queue depth."
            )

        if pdr < 0.65:
            issues.append(
                f"[OVERALL PDR = {pdr:.3f}] Still far from the 0.80 target. "
                f"The primary lever is jammer avoidance — fix that first."
            )

        return "\n".join(issues) if issues else (
            f"PDR = {pdr:.3f} — approaching target. Fine-tune energy and throughput consistency."
        )

    def _build_prompt(self, training_summary: str, current_code: str = None,
                      metrics: dict = None) -> str:
        """Build the full structured prompt for reward generation."""

        # ---- Section 1: Role & Domain Context ----------------------------
        role_context = """\
You are a senior reinforcement learning researcher and wireless communications expert.
Your job is to write a Python reward shaping function that teaches a tactical radio agent
to avoid enemy jammers, maintain high packet delivery rates, and stop unnecessary channel switching.

This is for a MILCOM research paper. The reward function you write will be benchmarked against
random, heuristic, and manually-shaped baselines. Your function MUST produce measurably better PDR
(Packet Delivery Ratio) than the current version. Previous versions failed — read them carefully below.
"""

        # ---- Section 2: Environment description ---------------------------
        env_context = """\
## Environment: Tactical Wireless Anti-Jamming Channel Selection

The agent selects 1 of 8 channels each timestep. A jammer uses one of three adaptive strategies
(changes every 100 steps), so the agent must continuously adapt:
  - RANDOM jammer  : randomly jams 2 channels per step
  - SWEEP jammer   : sweeps a 2-channel interference band across all channels
  - REACTIVE jammer: jams the agent's LAST used channel with 95% probability (+1 random channel)

Key insight: With 8 channels and 2 jammed at once, 6/8 = 75% of channels are clear at any step.
A perfect agent that always picks a clear channel achieves PDR ~ 0.85 (SNR failures cause the rest).
The agent's PDR is currently stuck at 0.62-0.65 — meaning it is wasting ~15-20% of transmissions
on jammed or sub-optimal channels. This is a solvable problem.

## State Variables (available in `state` dict)
| Variable                      | Type     | Range / Values     | Meaning                              |
|-------------------------------|----------|--------------------|--------------------------------------|
| state["channel_snr"]          | float[8] | ~0–25 dB           | Per-channel signal-to-noise ratio    |
| state["channel_success_rates"]| float[8] | 0.0–1.0            | Rolling packet success rate          |
| state["jammed_decay"]         | float[8] | 0.0–1.0            | 1.0 = very recently jammed, decays   |
| state["prev_channel"]         | int      | 0–7                | Last channel chosen (-1 at start)    |
| state["queue_length"]         | int      | 0–20               | Current packet backlog               |
| state["jammer_mode"]          | str      | random/sweep/reactv| Active jammer strategy               |
| state["ep_delivered"]         | int      | 0–500              | Packets delivered this episode       |
| state["ep_jammed_tx"]         | int      | 0–500              | Jammed TXs this episode              |
| state["ep_total_tx"]          | int      | 0–500              | Total TX attempts this episode       |
| state["step"]                 | int      | 0–499              | Current step                         |

## Feature Variables (pre-computed, available in `features` dict)
| Variable                        | Type  | Meaning                                            |
|---------------------------------|-------|----------------------------------------------------|
| features["switched"]            | bool  | Agent changed channel this step                    |
| features["current_ch_jammed"]   | bool  | Current channel has high jammed_decay (> 0.5)      |
| features["tx_success"]          | bool  | Packet delivered this step                         |
| features["on_best_channel"]     | bool  | Agent is on the highest-success-rate channel       |
| features["best_channel_success_rate"] | float | Success rate of the best available channel    |
| features["current_ch_success_rate"]   | float | Success rate of chosen channel                |
| features["queue_pressure"]      | float | queue_length / 20                                  |
| features["queue_critical"]      | bool  | queue_pressure > 0.80                              |
| features["running_pdr"]         | float | PDR so far this episode                            |
| features["running_jammed_rate"] | float | Jammed rate so far this episode                    |
| features["pdr_improving"]       | bool  | PDR higher than last step                          |

## Base Sparse Reward (env_reward)
  +1.0  packet delivered (unjammed, SNR OK)
  -1.0  transmission jammed
  -0.75 channel switch disrupted current packet
  -0.5  failed due to low SNR (not jammed)
"""

        # ---- Section 3: What failed so far --------------------------------
        failure_context = f"""
## CRITICAL: What Previous Reward Versions Tried and FAILED to Fix

{self.history.get_trend_summary()}

{self.history.get_failed_strategies()}

## DIAGNOSIS OF CURRENT FAILURE MODE
{self._diagnose_primary_failure(metrics)}
"""

        # ---- Section 4: Current reward code and performance ---------------
        if current_code:
            code_section = f"""
## Current (FAILING) Reward Function
The function below is currently active. It is NOT achieving the 0.80 PDR target.
Read it carefully and understand WHY it is failing before writing a replacement.

```python
{current_code}
```
"""
        else:
            code_section = """
## Current Reward Function
This is the first iteration — no reward function exists yet.
"""

        perf_section = f"""
## Most Recent Training Performance (20 evaluation episodes)
{training_summary}
"""

        # ---- Section 5: Precise instructions ------------------------------
        instruction = """
## Your Task: Write a BETTER Reward Function

You MUST fix the specific failure modes diagnosed above. A small incremental tweak is NOT enough
if the diagnosis shows a plateau — you must try a fundamentally different strategy.

### MANDATORY RULES (breaking any of these will crash training):
1. Function name: `compute_reward`
2. Signature: `def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float`
3. Always start with `reward = env_reward` — preserve the base signal
4. Import nothing inside compute_reward (numpy is pre-imported as `np` if needed)
5. Return type: float

### REWARD MAGNITUDE CALIBRATION TABLE (use this exactly):
The base reward is ±1.0 per step. Shaping must be significant enough to change behavior
but not so large that it overwrites the real signal.

| Severity    | Penalty/Bonus | When to use                                        |
|-------------|---------------|----------------------------------------------------|
| Critical    | ±0.20–0.30    | Agent consistently picks jammed channels (jammed > 0.25) |
| High        | ±0.10–0.20    | Persistent wrong behavior that hasn't improved     |
| Medium      | ±0.05–0.10    | Secondary issue, partially improving               |
| Low         | ±0.01–0.05    | Fine-tuning already-good behavior                  |

Previous runs used magnitudes of 0.01–0.08 for switching penalties and they FAILED.
If switching rate > 0.40, use a switching penalty of at least 0.15 per switch.
If jammed rate > 0.20, use a jammed-channel penalty of at least 0.20.

### ANTI-SWITCHING STRATEGY (critical — previous runs ignored this):
The agent switches ~50% of steps because the BENEFIT of switching (finding a clear channel)
outweighs the small switching penalty (0.01–0.05). Fix this with ONE OF:
  a) Strong stay-bonus: if NOT switched AND current channel is clean → +0.15 to +0.25
  b) Strong switch-penalty: if switched AND current channel was already clean → -0.20 to -0.30
  c) Momentum reward: reward proportional to consecutive steps on same clean channel

### ANTI-JAMMING STRATEGY (if jammed rate > 0.20):
  - Per-step penalty for any jammed TX: -0.20 to -0.30 (on top of env_reward's -1.0)
  - Additional penalty if agent STAYED on a jammed channel (didn't switch): -0.10 to -0.15
  - Bonus for switching AWAY from a jammed channel: +0.10 to +0.15

### OUTPUT FORMAT:
Write detailed reasoning FIRST (5+ paragraphs), then the code block.
Structure your reasoning as:
1. What the data shows (specific numbers)
2. Why the PREVIOUS reward failed (exact mechanism)
3. What specific change you are making (named strategy)
4. Exact magnitude choices and calibration reasoning
5. Expected behavior change in next iteration

```python
def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    \"\"\"
    Anti-Jamming Reward v[N] — [one-line strategy description]
    Key changes from previous: [list what's different]
    \"\"\"
    reward = env_reward  # Base: +1.0 delivery, -1.0 jammed, -0.75 switch-disrupt, -0.5 SNR-fail

    # --- [Strategy Name] ---
    # [comment explaining why this magnitude was chosen]

    return reward
```
"""

        return (role_context + env_context + failure_context +
                code_section + perf_section + instruction)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Extract code block and reasoning from LLM response."""

        # Extract reasoning
        code_split = response.split('```python')
        reasoning = code_split[0].strip() if len(code_split) > 1 else "No reasoning provided"

        # Extract code
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if 'def compute_reward' not in code:
                code = self._get_fallback_reward()
                reasoning = (f"LLM output had no compute_reward function — "
                             f"using robust fallback. Raw: {response[:200]}")
        else:
            code = self._get_fallback_reward()
            reasoning = f"Could not parse code block from LLM — using fallback. Raw: {response[:200]}"

        # Safety: clamp any accidentally-large magnitudes
        code = self._safety_check_code(code)
        return code, reasoning

    def _safety_check_code(self, code: str) -> str:
        """
        Warn in the code if magnitudes > 0.5 appear — don't auto-clamp so as
        not to silently change LLM intent, but catch obvious errors.
        """
        # Check for obviously dangerous literals (>= 1.0 magnitude bonus/penalty)
        dangerous = re.findall(r'[+-]\s*([1-9]\d*\.?\d*|0\.[5-9]\d*)\b', code)
        if dangerous:
            code = (
                "# WARNING: Large magnitude values detected — verify they are intentional.\n"
                "# Values found: " + str(dangerous[:5]) + "\n" + code
            )
        return code

    # ------------------------------------------------------------------
    # Fallback reward (used when LLM fails)
    # ------------------------------------------------------------------

    def _get_fallback_reward(self) -> str:
        """Robust fallback reward — calibrated from Colab run failure analysis."""
        return '''def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """
    Fallback Anti-Jamming Reward — Aggressive Version
    Based on analysis: previous small-magnitude rewards failed to reduce switching.
    Uses larger magnitudes to drive behavior change.
    """
    import numpy as np
    reward = env_reward  # +1.0 delivery / -1.0 jammed / -0.75 switch-disrupt / -0.5 SNR fail

    prev_ch   = state.get("prev_channel", -1)
    jammed_d  = state.get("jammed_decay", [0.0] * 8)
    succ_r    = state.get("channel_success_rates", [0.5] * 8)
    n_ch      = state.get("n_channels", 8)

    # ---- [1] Strong jammed-channel penalty --------------------------------
    # If agent selected a recently-jammed channel, penalise hard.
    # Magnitude: -0.25 (was 0.05 in failed version)
    if features.get("current_ch_jammed", False):
        reward -= 0.25

    # ---- [2] Stay-on-good-channel bonus ------------------------------------
    # Reward STAYING on a clean, high-success channel — counteracts thrashing
    switched = features.get("switched", False)
    ch_ok    = not features.get("current_ch_jammed", False)
    ch_succ  = features.get("current_ch_success_rate", 0.5)
    if not switched and ch_ok and ch_succ > 0.60:
        reward += 0.20  # Strong stay-bonus to compete with switch urge

    # ---- [3] Unnecessary switching penalty --------------------------------
    # Only penalise switching away from a GOOD channel (not forced escape)
    if switched:
        if 0 <= prev_ch < n_ch:
            prev_jammed  = jammed_d[prev_ch] > 0.5
            prev_success = succ_r[prev_ch]
            if not prev_jammed and prev_success > 0.55:
                # Switched away from a clean channel — punish
                reward -= 0.25
            else:
                # Switched AWAY from a bad channel — small bonus for escaping
                reward += 0.05

    # ---- [4] Queue buildup penalty ----------------------------------------
    queue_pressure = features.get("queue_pressure", 0.0)
    if queue_pressure > 0.80:
        reward -= 0.15
    elif queue_pressure > 0.60:
        reward -= 0.08

    # ---- [5] Jammer-escape bonus ------------------------------------------
    # If reactive jammer was targeting previous channel and agent escaped → good
    jammer_mode = state.get("jammer_mode", "random")
    if switched and jammer_mode == "reactive":
        if 0 <= prev_ch < n_ch and jammed_d[prev_ch] > 0.7:
            reward += 0.10  # Correct escape from reactive jammer

    return reward
'''

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _save_to_history(self, code: str, reasoning: str, training_summary: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = REWARD_HISTORY_DIR / f"reward_v{self.iteration}_{timestamp}.py"
        header = f'''"""
Anti-Jamming Reward Function v{self.iteration}
Generated: {datetime.now().isoformat()}
Model: {self.model}

REASONING:
{reasoning[:2000]}

TRAINING SUMMARY:
{training_summary[:1000]}
"""

'''
        with open(filename, 'w') as f:
            f.write(header + code)

    def save_current_reward(self, code: str):
        with open(REWARD_FILE, 'w') as f:
            f.write(code)

    def load_current_reward(self) -> Optional[str]:
        if REWARD_FILE.exists():
            return REWARD_FILE.read_text()
        return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_ollama_connection(host: str = None) -> bool:
    host = host or LLM_CONFIG["host"]
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models(host: str = None) -> list:
    host = host or LLM_CONFIG["host"]
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
        return []
    except:
        return []


if __name__ == "__main__":
    print("Testing Ollama connection...")
    if test_ollama_connection():
        print("Ollama is running!")
        print(f"Available models: {list_available_models()}")

        generator = LLMRewardGenerator()
        test_summary = """## Training Results — Last 20 Episodes
- Average PDR: 0.63
- Average Jammed TX Rate: 0.22
- Average Switching Rate: 0.56/step
- Average Queue Length: 11.4 packets

### Observed Behaviors
- [OK] Moderate PDR — partially avoids jammer
- [HIGH] Jammed TX rate 22% — still selecting jammed channels too often
- [SWITCH-HIGH] Switching 56% of steps — reward too weak to stop thrashing
"""
        print("\nGenerating test reward function...")
        try:
            code, reasoning = generator.generate_reward(
                test_summary,
                metrics={"avg_pdr": 0.63, "avg_jammed_rate": 0.22, "avg_switch_rate": 0.56}
            )
            print(f"\nReasoning (first 500 chars):\n{reasoning[:500]}")
            print(f"\nGenerated Code:\n{code}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("ERROR: Ollama is not running! Start it with: ollama serve")
