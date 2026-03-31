"""LLM Interface for Reward Function Generation via Ollama"""
import json
import re
import requests
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import LLM_CONFIG, REWARD_FILE, REWARD_HISTORY_DIR


class LLMRewardGenerator:
    """
    Interfaces with local LLM (Ollama) to generate and refine reward functions.
    """
    
    def __init__(self, model: str = None, host: str = None):
        self.model = model or LLM_CONFIG["model"]
        self.host = host or LLM_CONFIG["host"]
        self.temperature = LLM_CONFIG.get("temperature", 0.8)
        self.num_ctx = LLM_CONFIG.get("num_ctx", 8192)
        self.iteration = 0
    
    def _call_ollama(self, prompt: str) -> str:
        """Make a request to Ollama API"""
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
            raise TimeoutError("Ollama request timed out after 120 seconds")
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
        """Build the prompt for reward generation"""
        
        base_context = """You are an expert reinforcement learning engineer optimizing a reward function for Atari Pong.

## Game Context
- Player controls the RIGHT paddle
- Goal: Return the ball past the CPU (left) paddle to score
- Score range: -21 (complete loss) to +21 (complete win)
- Ball position: x (0-160), y (0-210)
- Paddle position: y (0-210)

## Available State Variables (in compute_reward function)
- state["ball_x"]: Ball x position (0-160)
- state["ball_y"]: Ball y position (0-210) 
- state["player_paddle_y"]: Your paddle y position (0-210)
- state["cpu_paddle_y"]: CPU paddle y position
- state["player_score"]: Your current score
- state["cpu_score"]: CPU's current score
- features["paddle_ball_distance_y"]: Distance between paddle and ball on Y axis
- features["paddle_aligned"]: Boolean, True if paddle within 10px of ball
- features["ball_approaching"]: Boolean, True if ball coming toward player
- features["moving_toward_ball"]: Boolean, True if paddle moving toward ball
- features["ball_velocity_x"]: Ball x velocity
- features["ball_velocity_y"]: Ball y velocity  
- features["paddle_velocity"]: Paddle movement velocity
- env_reward: The original game reward (+1 for scoring, -1 for opponent scoring)
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
None - this is the initial reward function.
"""
        
        training_section = f"""
## Recent Training Performance
{training_summary}
"""
        
        instruction = """
## Your Task
Based on the training performance above, generate an improved reward function.

IMPORTANT GUIDELINES:
1. The function must be named `compute_reward`
2. Signature: `def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float`
3. Keep env_reward as a base - it provides score signals
4. Add shaping rewards that are SMALL (0.001 to 0.1 scale) to guide behavior
5. Focus on ONE or TWO key improvements at a time
6. Reward shaping should guide, not dominate the learning signal

COMMON REWARD SHAPING STRATEGIES:
- Reward staying close to ball Y position (alignment)
- Reward moving toward the ball when it approaches
- Small penalty for being far from ball
- Reward hitting the ball (when ball velocity reverses)
- Penalty for unnecessary movement when ball is far

OUTPUT FORMAT:
First, explain your detailed reasoning. Include: what the agent is doing wrong based on the metrics, why the current reward function fails to address it, what specific change you are making, why you chose these specific magnitudes, and what behavior improvement you expect. Be thorough.
Then provide the code in a ```python code block.

Example output format:
REASONING: The agent is losing because... I will add...

```python
def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    reward = env_reward  # Base game reward
    
    # Your shaping logic here
    
    return reward
```
"""
        
        return base_context + current_section + training_section + instruction
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract code and reasoning"""
        
        # Extract reasoning (everything before the code block)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=```python|```)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Try to get any text before code block
            pre_code = response.split('```python')[0] if '```python' in response else ""
            reasoning = pre_code.strip() if pre_code else "No reasoning provided"
        
        # Extract code block
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            
            # Validate it has the right function
            if 'def compute_reward' not in code:
                code = self._get_fallback_reward()
                reasoning = "LLM output invalid - using fallback reward"
        else:
            code = self._get_fallback_reward()
            reasoning = "Could not parse LLM response - using fallback reward"
        
        return code, reasoning
    
    def _get_fallback_reward(self) -> str:
        """Return a basic fallback reward function"""
        return '''def compute_reward(state: dict, features: dict, env_reward: float, done: bool) -> float:
    """Basic reward with paddle-ball alignment shaping"""
    reward = env_reward  # Base game reward
    
    # Reward staying aligned with ball
    distance = features.get("paddle_ball_distance_y", 50)
    alignment_reward = max(0, 1 - distance / 100) * 0.01
    reward += alignment_reward
    
    # Small reward for moving toward ball when it approaches
    if features.get("ball_approaching", False) and features.get("moving_toward_ball", False):
        reward += 0.005
    
    return reward
'''
    
    def _save_to_history(self, code: str, reasoning: str, training_summary: str):
        """Save generated reward to history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = REWARD_HISTORY_DIR / f"reward_v{self.iteration}_{timestamp}.py"
        
        header = f'''"""
Reward Function v{self.iteration}
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
        """Save reward function as the current active reward"""
        with open(REWARD_FILE, 'w') as f:
            f.write(code)
    
    def load_current_reward(self) -> Optional[str]:
        """Load the current reward function code"""
        if REWARD_FILE.exists():
            return REWARD_FILE.read_text()
        return None


def test_ollama_connection(host: str = None) -> bool:
    """Test if Ollama is running and accessible"""
    host = host or LLM_CONFIG["host"]
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models(host: str = None) -> list:
    """List models available in Ollama"""
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
        
        # Test generation
        generator = LLMRewardGenerator()
        test_summary = """## Training Results (Last 10 Episodes)
- Average Score: -15.0
- Win Rate: 0%
- Average Paddle-Ball Distance: 65.0 pixels

### Observed Behaviors
- Agent is losing badly - not tracking ball effectively
- Paddle very far from ball (>60px)
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
