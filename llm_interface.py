"""
LLM Interface for Reward Function Generation — Anti-Jamming Channel Selection
=============================================================================
Interfaces with a local LLM (Ollama) to iteratively generate and refine
reward functions for tactical wireless anti-jamming channel selection.

The domain context, state variable descriptions, and safety constraints
are all framed around the wireless communication problem, not Pong.
"""
import json
import re
import requests
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import LLM_CONFIG, REWARD_FILE, REWARD_HISTORY_DIR


class LLMRewardGenerator:
    """
    Interfaces with local LLM (Ollama) to generate and refine reward functions
    for the anti-jamming channel selection task.
    """

    def __init__(self, model: str = None, host: str = None):
        self.model = model or LLM_CONFIG["model"]
        self.host = host or LLM_CONFIG["host"]
        self.temperature = LLM_CONFIG.get("temperature", 0.7)
        self.num_ctx = LLM_CONFIG.get("num_ctx", 8192)
        self.iteration = 0

    def _call_ollama(self, prompt: str) -> str:
        """Make a request to Ollama API."""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=500)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out after 500 seconds")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate_reward(self, training_summary: str, current_reward_code: str = None) -> Tuple[str, str]:
        """
        Generate or refine a reward function based on training performance.

        Args:
            training_summary: Analysis of recent training episodes
            current_reward_code: Current reward function code (if refining)

        Returns:
            Tuple of (reward_function_code, reasoning)
        """
        self.iteration += 1

        prompt = self._build_prompt(training_summary, current_reward_code)
        response = self._call_ollama(prompt)

        code, reasoning = self._parse_response(response)

        # Save to history
        self._save_to_history(code, reasoning, training_summary)

        return code, reasoning

    def _build_prompt(self, training_summary: str, current_code: str = None) -> str:
        """Build the prompt for reward generation — wireless domain."""

        base_context = """You are an expert reinforcement learning engineer designing reward functions
for a tactical wireless anti-jamming channel selection agent in a military communications scenario.

## Task Context
A cognitive radio agent must select one of N wireless channels at each timestep to transmit packets.
A jammer attacks channels using one of three strategies that can change during an episode:
  - Random Jammer: jams a random channel each step
  - Sweep Jammer: sweeps through channels sequentially
  - Reactive Jammer: jams whichever channel the agent used last

The agent succeeds when it delivers packets by selecting unjammed channels with sufficient SNR.

## Available State Variables (in `state` dict passed to compute_reward)
- state["channel_snr"]           : numpy array[N] — per-channel SNR in dB (raw values ~0-25)
- state["channel_success_rates"] : numpy array[N] — rolling success rate per channel (0.0–1.0)
- state["jammed_decay"]          : numpy array[N] — decaying jammer indicator per channel (0=clear, 1=recently jammed)
- state["prev_channel"]          : int            — channel selected in previous step (-1 if start)
- state["queue_length"]          : int            — current packet queue depth (0–20)
- state["energy_used"]           : float          — cumulative energy used this episode
- state["jammer_mode"]           : str            — current jammer mode: "random", "sweep", or "reactive"
- state["step"]                  : int            — current step number
- state["max_steps"]             : int            — episode length (500)
- state["n_channels"]            : int            — number of channels (8)
- state["ep_delivered"]          : int            — packets delivered so far this episode
- state["ep_total_tx"]           : int            — total transmission attempts so far
- state["ep_jammed_tx"]          : int            — total jammed transmissions so far

## Available Feature Variables (in `features` dict passed to compute_reward)
- features["switched"]               : bool  — agent switched channel this step
- features["is_jammed"]              : bool  — selected channel was jammed this step
- features["tx_success"]             : bool  — packet was delivered this step
- features["avoided_jammed"]         : bool  — selected channel was NOT in jammed set
- features["queue_pressure"]         : float — queue/capacity ratio (0.0–1.0)
- features["queue_critical"]         : bool  — queue > 80% capacity
- features["running_pdr"]            : float — packet delivery ratio so far this episode
- features["running_jammed_rate"]    : float — jammed TX rate so far this episode
- features["best_channel"]           : int   — channel with highest recent success rate
- features["best_channel_success_rate"]: float — success rate of best channel
- features["current_ch_jammed"]      : bool  — current channel has high jammed_decay
- features["current_ch_snr"]         : float — SNR of current channel (dB)
- features["current_ch_success_rate"]: float — rolling success rate of current channel
- features["on_best_channel"]        : bool  — agent is on the best channel
- features["switch_penalty_active"]  : bool  — channel switch occurred (same as "switched")
- features["throughput_so_far"]      : float — delivered packets / steps so far
- features["pdr_improving"]          : bool  — PDR is higher than previous step
- features["energy_budget_used"]     : float — average energy per step so far

## env_reward (sparse base signal)
- env_reward = +1.0  if packet delivered (not jammed AND SNR OK)
- env_reward = -1.0  if transmission was jammed
- env_reward = -0.5  if transmission failed due to low SNR (not jammed)
"""

        if current_code:
            current_section = f"""
## Current Reward Function
```python
{current_code}
```
"""
        else:
            current_section = """
## Current Reward Function
None — this is the initial reward function generation.
"""

        training_section = f"""
## Recent Training Performance
{training_summary}
"""

        instruction = """
## Your Task
Based on the training performance above, generate an improved reward function that helps the agent:
1. Avoid jammed channels (primary objective)
2. Maintain high packet delivery ratio (PDR)
3. Reduce unnecessary channel switching
4. Manage queue buildup
5. Use energy efficiently

CRITICAL RULES (VIOLATING THESE WILL BREAK TRAINING):
1. The function MUST be named `compute_reward`
2. Signature: `def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float`
3. Keep env_reward as the base — it provides the real sparse signal (+1/-1/-0.5)
4. All shaping reward magnitudes MUST be between 0.001 and 0.1. NEVER use 0.2, 0.5, or 1.0.
5. Focus on ONE or TWO improvements at a time. Small, incremental changes only.
6. The total shaped addition per step MUST NOT exceed ±0.15. Shaping GUIDES, not DOMINATES.
7. Never import external libraries inside compute_reward — only use standard Python and numpy if needed.

MAGNITUDE REFERENCE (follow strictly):
- Good values: 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1
- BAD values that will crash training: 0.2, 0.25, 0.5, 1.0, 2.0

WIRELESS-SPECIFIC SHAPING STRATEGIES:
- Penalty for selecting a recently-jammed channel: -0.05 to -0.10
- Reward for selecting a high-success-rate channel: +0.01 to +0.05
- Penalty for excessive channel switching: -0.02 to -0.05
- Penalty for queue buildup when critical: -0.02 to -0.08
- Reward for staying on a clean channel when PDR is high: +0.01 to +0.03
- Penalty for transmitting on jammed channel repeatedly: -0.05 to -0.10

OUTPUT FORMAT:
First provide detailed reasoning in multiple paragraphs:
- What the agent is doing wrong based on the wireless metrics
- Why the current reward fails to address it
- What specific change you're making and why
- Why you chose these specific magnitudes (must be 0.001–0.1 range)
- What behavior improvement you expect

Then provide the code in a ```python code block.

REASONING: [your detailed multi-paragraph analysis here]

```python
def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base sparse reward: +1 delivery, -1 jammed, -0.5 SNR fail
    
    # Your wireless-domain shaping logic here (all values between 0.001 and 0.1)
    
    return reward
```
"""

        return base_context + current_section + training_section + instruction

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract code and reasoning."""

        # Extract reasoning (everything before the code block)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=```python|```)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            pre_code = response.split('```python')[0] if '```python' in response else ""
            reasoning = pre_code.strip() if pre_code else "No reasoning provided"

        # Extract code block
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)

        if code_match:
            code = code_match.group(1).strip()

            # Validate it has the right function
            if 'def compute_reward' not in code:
                code = self._get_fallback_reward()
                reasoning = "LLM output invalid (no compute_reward function found) — using fallback reward"
        else:
            code = self._get_fallback_reward()
            reasoning = "Could not parse LLM response — using fallback reward"

        return code, reasoning

    def _get_fallback_reward(self) -> str:
        """Return a basic fallback wireless reward function."""
        return '''def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """
    Fallback wireless reward: penalise jammed-channel selection,
    reward good channels, small penalty for switching.
    """
    reward = env_reward  # +1.0 delivery / -1.0 jammed / -0.5 SNR fail

    # Penalty for selecting a recently-jammed channel
    if features.get("current_ch_jammed", False):
        reward -= 0.05

    # Reward for being on the best available channel
    if features.get("on_best_channel", False) and not features.get("current_ch_jammed", False):
        reward += 0.02

    # Penalty for excessive channel switching
    if features.get("switched", False):
        reward -= 0.01

    # Penalty for severe queue buildup
    if features.get("queue_critical", False):
        reward -= 0.03

    return reward
'''

    def _save_to_history(self, code: str, reasoning: str, training_summary: str):
        """Save generated reward to history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = REWARD_HISTORY_DIR / f"reward_v{self.iteration}_{timestamp}.py"

        header = f'''"""
Anti-Jamming Reward Function v{self.iteration}
Generated: {datetime.now().isoformat()}

REASONING:
{reasoning}

TRAINING SUMMARY:
{training_summary}
"""

'''

        with open(filename, 'w') as f:
            f.write(header + code)

    def save_current_reward(self, code: str):
        """Save reward function as the current active reward."""
        with open(REWARD_FILE, 'w') as f:
            f.write(code)

    def load_current_reward(self) -> Optional[str]:
        """Load the current reward function code."""
        if REWARD_FILE.exists():
            return REWARD_FILE.read_text()
        return None


def test_ollama_connection(host: str = None) -> bool:
    """Test if Ollama is running and accessible."""
    host = host or LLM_CONFIG["host"]
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models(host: str = None) -> list:
    """List models available in Ollama."""
    host = host or LLM_CONFIG["host"]
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        return []
    except:
        return []


if __name__ == "__main__":
    # Test the interface
    print("Testing Ollama connection...")

    if test_ollama_connection():
        print("Ollama is running!")
        models = list_available_models()
        print(f"Available models: {models}")

        generator = LLMRewardGenerator()
        test_summary = """## Training Results — Anti-Jamming Channel Selection (Last 10 Episodes)
- Average PDR: 0.32
- Average Jammed TX Rate: 0.45
- Average Switching Rate: 0.08/step
- Average Queue Length: 12.3 packets

### Observed Behaviors
- Agent is selecting jammed channels frequently
- Queue is building up — throughput insufficient
"""

        print("\nGenerating test reward function...")
        try:
            code, reasoning = generator.generate_reward(test_summary)
            print(f"\nReasoning: {reasoning}")
            print(f"\nGenerated Code:\n{code}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("ERROR: Ollama is not running!")
        print("Start it with: ollama serve")
