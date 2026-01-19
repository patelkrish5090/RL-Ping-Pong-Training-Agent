"""Gymnasium Wrapper for Dynamic Reward Shaping in Pong"""
import importlib.util
import gymnasium as gym
import numpy as np
from typing import Callable, Dict, Optional, Any
from pathlib import Path

from state_extractor import extract_pong_state, compute_derived_features
from config import REWARD_FILE


class PongRewardWrapper(gym.Wrapper):
    """
    Wrapper that applies dynamic reward shaping to Pong.
    
    The reward function can be hot-swapped during training by updating
    the reward file and calling reload_reward().
    """
    
    def __init__(self, env: gym.Env, reward_file: Path = None):
        super().__init__(env)
        self.reward_file = reward_file or REWARD_FILE
        self.compute_reward: Optional[Callable] = None
        self.prev_state: Optional[Dict] = None
        
        # Episode tracking
        self.episode_hits = 0
        self.episode_misses = 0
        self.episode_distances = []
        self.prev_player_score = 0
        self.prev_cpu_score = 0
        
        # Load initial reward function
        self.reload_reward()
    
    def reload_reward(self) -> bool:
        """
        Reload the reward function from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.reward_file.exists():
                print(f"Warning: Reward file not found at {self.reward_file}")
                self.compute_reward = self._default_reward
                return False
            
            # Dynamic import of reward module
            spec = importlib.util.spec_from_file_location("reward_module", self.reward_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'compute_reward'):
                self.compute_reward = module.compute_reward
                return True
            else:
                print("Warning: No compute_reward function found in reward file")
                self.compute_reward = self._default_reward
                return False
                
        except Exception as e:
            print(f"Error loading reward function: {e}")
            self.compute_reward = self._default_reward
            return False
    
    def _default_reward(self, state: dict, features: dict, env_reward: float, done: bool) -> float:
        """Fallback reward: just return environment reward"""
        return env_reward
    
    def reset(self, **kwargs) -> tuple:
        """Reset environment and episode tracking"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode stats
        self.prev_state = None
        self.episode_hits = 0
        self.episode_misses = 0
        self.episode_distances = []
        self.prev_player_score = 0
        self.prev_cpu_score = 0
        
        return obs, info
    
    def step(self, action) -> tuple:
        """Execute step and apply reward shaping"""
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Extract state from RAM
        try:
            ram = self.env.unwrapped.ale.getRAM()
            state = extract_pong_state(ram)
            features = compute_derived_features(state, self.prev_state)
        except Exception:
            # If RAM extraction fails, use empty state
            state = {}
            features = {}
        
        # Track hits and misses based on score changes
        player_score = state.get("player_score", 0)
        cpu_score = state.get("cpu_score", 0)
        
        if player_score > self.prev_player_score:
            self.episode_hits += 1  # We scored
        if cpu_score > self.prev_cpu_score:
            self.episode_misses += 1  # Opponent scored
        
        self.prev_player_score = player_score
        self.prev_cpu_score = cpu_score
        
        # Track paddle-ball distance
        distance = features.get("paddle_ball_distance_y", 50)
        self.episode_distances.append(distance)
        
        # Compute shaped reward
        done = terminated or truncated
        if self.compute_reward:
            try:
                shaped_reward = self.compute_reward(state, features, env_reward, done)
            except Exception as e:
                print(f"Reward computation error: {e}")
                shaped_reward = env_reward
        else:
            shaped_reward = env_reward
        
        # Update state tracking
        self.prev_state = state
        
        # Add episode stats to info
        info["state"] = state
        info["features"] = features
        info["env_reward"] = env_reward
        info["shaped_reward"] = shaped_reward
        
        return obs, shaped_reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for the completed episode"""
        return {
            "hits": self.episode_hits,
            "misses": self.episode_misses,
            "paddle_distances": self.episode_distances.copy(),
            "avg_distance": np.mean(self.episode_distances) if self.episode_distances else 50.0,
        }


class VecPongRewardWrapper:
    """
    Wrapper for vectorized environments to apply reward shaping.
    Works with SB3's VecEnv interface.
    """
    
    def __init__(self, venv, reward_file: Path = None):
        self.venv = venv
        self.reward_file = reward_file or REWARD_FILE
        self.compute_reward: Optional[Callable] = None
        self.n_envs = venv.num_envs
        
        # Per-env state tracking
        self.prev_states = [None] * self.n_envs
        self.prev_player_scores = [0] * self.n_envs
        self.prev_cpu_scores = [0] * self.n_envs
        
        # Episode stats tracking
        self.episode_hits = [0] * self.n_envs
        self.episode_misses = [0] * self.n_envs
        self.episode_distances = [[] for _ in range(self.n_envs)]
        
        # Completed episodes buffer
        self.completed_episodes = []
        
        self.reload_reward()
    
    def reload_reward(self) -> bool:
        """Reload reward function from file"""
        try:
            if not self.reward_file.exists():
                self.compute_reward = lambda s, f, r, d: r
                return False
            
            spec = importlib.util.spec_from_file_location("reward_module", self.reward_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'compute_reward'):
                self.compute_reward = module.compute_reward
                return True
            else:
                self.compute_reward = lambda s, f, r, d: r
                return False
                
        except Exception as e:
            print(f"Error loading reward: {e}")
            self.compute_reward = lambda s, f, r, d: r
            return False
    
    def step_wait(self):
        """Process step results and apply reward shaping"""
        obs, rewards, dones, infos = self.venv.step_wait()
        
        shaped_rewards = np.array(rewards, dtype=np.float32)
        
        for i in range(self.n_envs):
            try:
                # Get RAM from individual env
                env = self.venv.envs[i] if hasattr(self.venv, 'envs') else self.venv.venv.envs[i]
                ram = env.unwrapped.ale.getRAM()
                state = extract_pong_state(ram)
                features = compute_derived_features(state, self.prev_states[i])
                
                # Track scores
                player_score = state.get("player_score", 0)
                cpu_score = state.get("cpu_score", 0)
                
                if player_score > self.prev_player_scores[i]:
                    self.episode_hits[i] += 1
                if cpu_score > self.prev_cpu_scores[i]:
                    self.episode_misses[i] += 1
                
                self.prev_player_scores[i] = player_score
                self.prev_cpu_scores[i] = cpu_score
                
                # Track distance
                distance = features.get("paddle_ball_distance_y", 50)
                self.episode_distances[i].append(distance)
                
                # Compute shaped reward
                if self.compute_reward:
                    shaped_rewards[i] = self.compute_reward(state, features, rewards[i], dones[i])
                
                self.prev_states[i] = state
                
            except Exception:
                pass  # Keep original reward on error
            
            # Handle episode end
            if dones[i]:
                self.completed_episodes.append({
                    "hits": self.episode_hits[i],
                    "misses": self.episode_misses[i],
                    "paddle_distances": self.episode_distances[i].copy(),
                })
                # Reset tracking for this env
                self.episode_hits[i] = 0
                self.episode_misses[i] = 0
                self.episode_distances[i] = []
                self.prev_states[i] = None
                self.prev_player_scores[i] = 0
                self.prev_cpu_scores[i] = 0
        
        return obs, shaped_rewards, dones, infos
    
    def step_async(self, actions):
        """Forward to wrapped env"""
        self.venv.step_async(actions)
    
    def step(self, actions):
        """Full step with reward shaping"""
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        """Reset all envs"""
        self.prev_states = [None] * self.n_envs
        self.prev_player_scores = [0] * self.n_envs
        self.prev_cpu_scores = [0] * self.n_envs
        self.episode_hits = [0] * self.n_envs
        self.episode_misses = [0] * self.n_envs
        self.episode_distances = [[] for _ in range(self.n_envs)]
        return self.venv.reset()
    
    def get_completed_episodes(self, clear: bool = True) -> list:
        """Get completed episode stats, optionally clearing buffer"""
        episodes = self.completed_episodes.copy()
        if clear:
            self.completed_episodes = []
        return episodes
    
    # Forward other methods to wrapped env
    def __getattr__(self, name):
        return getattr(self.venv, name)


def make_pong_env(reward_file: Path = None) -> gym.Env:
    """
    Create a Pong environment with reward shaping wrapper.
    
    Args:
        reward_file: Path to reward function file
    
    Returns:
        Wrapped Pong environment
    """
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    
    env = gym.make("PongNoFrameskip-v4")
    env = AtariWrapper(env)
    env = PongRewardWrapper(env, reward_file)
    
    return env
