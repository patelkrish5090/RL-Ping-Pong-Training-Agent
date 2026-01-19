"""
Iterative Reward Refinement Training for Pong

Main training script that:
1. Trains PPO agent on Pong
2. Periodically analyzes performance
3. Uses LLM to refine reward function
4. Hot-swaps reward and continues training
"""
import argparse
import time
from datetime import datetime
from pathlib import Path
import shimmy
import ale_py


import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    ENV_ID, N_ENVS, FRAME_STACK, PPO_CONFIG,
    EPISODES_PER_UPDATE, MAX_ITERATIONS, TARGET_SCORE,
    TIMESTEPS_PER_ITERATION, LOG_DIR, MODEL_DIR, REWARD_FILE
)
from state_extractor import extract_pong_state, compute_derived_features
from episode_analyzer import EpisodeAnalyzer
from llm_interface import LLMRewardGenerator, test_ollama_connection


class RewardShapingCallback(BaseCallback):
    """
    Callback that tracks episode stats and applies reward shaping.
    Also handles hot-swapping of reward functions.
    """
    
    def __init__(self, analyzer: EpisodeAnalyzer, reward_file: Path, verbose: int = 0):
        super().__init__(verbose)
        self.analyzer = analyzer
        self.reward_file = reward_file
        self.compute_reward = None
        self.prev_states = {}  # Per-env state tracking
        self.prev_scores = {}  # Per-env score tracking
        self.episode_distances = {}  # Per-env distance tracking
        self.episode_hits = {}
        self.episode_misses = {}
        
        self.reload_reward()
    
    def reload_reward(self) -> bool:
        """Reload reward function from file"""
        try:
            import importlib.util
            if not self.reward_file.exists():
                print(f"Reward file not found: {self.reward_file}")
                return False
            
            spec = importlib.util.spec_from_file_location("reward_module", self.reward_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'compute_reward'):
                self.compute_reward = module.compute_reward
                print(f"Reloaded reward function from {self.reward_file}")
                return True
            return False
        except Exception as e:
            print(f"Error reloading reward: {e}")
            return False
    
    def _on_step(self) -> bool:
        """Called at each step - apply reward shaping"""
        # Get environments
        envs = self.training_env.envs if hasattr(self.training_env, 'envs') else None
        if envs is None:
            return True
        
        for i, env in enumerate(envs):
            env_id = i
            
            # Initialize tracking for new envs
            if env_id not in self.prev_states:
                self.prev_states[env_id] = None
                self.prev_scores[env_id] = (0, 0)
                self.episode_distances[env_id] = []
                self.episode_hits[env_id] = 0
                self.episode_misses[env_id] = 0
            
            try:
                # Extract state
                ale_env = env
                while hasattr(ale_env, 'env'):
                    ale_env = ale_env.env
                
                ram = ale_env.ale.getRAM()
                state = extract_pong_state(ram)
                features = compute_derived_features(state, self.prev_states[env_id])
                
                # Track score changes
                player_score = state.get("player_score", 0)
                cpu_score = state.get("cpu_score", 0)
                prev_player, prev_cpu = self.prev_scores[env_id]
                
                if player_score > prev_player:
                    self.episode_hits[env_id] += 1
                if cpu_score > prev_cpu:
                    self.episode_misses[env_id] += 1
                
                self.prev_scores[env_id] = (player_score, cpu_score)
                
                # Track distance
                distance = features.get("paddle_ball_distance_y", 50)
                self.episode_distances[env_id].append(distance)
                
                # Apply reward shaping
                if self.compute_reward is not None:
                    env_reward = self.locals.get('rewards', [0])[i] if 'rewards' in self.locals else 0
                    done = self.locals.get('dones', [False])[i] if 'dones' in self.locals else False
                    
                    try:
                        shaped = self.compute_reward(state, features, float(env_reward), done)
                        if 'rewards' in self.locals:
                            self.locals['rewards'][i] = shaped
                    except Exception:
                        pass
                
                self.prev_states[env_id] = state
                
            except Exception:
                pass
            
            # Check for episode end
            dones = self.locals.get('dones', None)
            if dones is not None and dones[i]:
                # Record episode stats
                infos = self.locals.get('infos', [{}])
                info = infos[i] if i < len(infos) else {}
                
                # Get episode reward from info
                ep_info = info.get('episode', {})
                ep_reward = ep_info.get('r', 0)
                ep_length = ep_info.get('l', 0)
                
                self.analyzer.add_episode({
                    'total_reward': ep_reward,
                    'length': ep_length,
                    'hits': self.episode_hits[env_id],
                    'misses': self.episode_misses[env_id],
                    'paddle_distances': self.episode_distances[env_id].copy(),
                })
                
                # Reset tracking
                self.prev_states[env_id] = None
                self.prev_scores[env_id] = (0, 0)
                self.episode_distances[env_id] = []
                self.episode_hits[env_id] = 0
                self.episode_misses[env_id] = 0
        
        return True


from stable_baselines3.common.monitor import Monitor

def make_env():
    """Create a single Pong environment with Atari wrappers"""
    env = gym.make(ENV_ID)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env


def make_vec_env(n_envs: int = N_ENVS):
    """Create vectorized Pong environments"""
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


def train_iterative(
    timesteps_per_iter: int = TIMESTEPS_PER_ITERATION,
    max_iterations: int = MAX_ITERATIONS,
    target_score: float = TARGET_SCORE,
    episodes_for_analysis: int = EPISODES_PER_UPDATE,
    use_llm: bool = True,
    verbose: int = 1,
):
    """
    Main iterative training loop.
    
    Args:
        timesteps_per_iter: Training timesteps between LLM updates
        max_iterations: Maximum number of LLM refinement iterations
        target_score: Target average score to achieve
        episodes_for_analysis: Number of episodes to analyze for LLM
        use_llm: Whether to use LLM for reward refinement
        verbose: Verbosity level
    """
    print("=" * 60)
    print("Iterative Reward Refinement Training for Pong")
    print("=" * 60)
    
    # Check Ollama if using LLM
    if use_llm:
        if not test_ollama_connection():
            print("\nWARNING: Ollama not running! Start with 'ollama serve'")
            print("Continuing without LLM refinement...\n")
            use_llm = False
        else:
            print("Ollama connection OK")
    
    # Initialize components
    analyzer = EpisodeAnalyzer()
    llm = LLMRewardGenerator() if use_llm else None
    
    # Create environments
    print(f"\nCreating {N_ENVS} parallel environments...")
    env = make_vec_env(N_ENVS)
    
    # Create callback
    callback = RewardShapingCallback(analyzer, REWARD_FILE, verbose=verbose)
    
    # Initialize PPO
    print("Initializing PPO agent...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=verbose,
        tensorboard_log=str(LOG_DIR),
        **PPO_CONFIG
    )
    
    # Training loop
    total_timesteps = 0
    best_score = -21
    
    print(f"\nStarting training...")
    print(f"  Timesteps per iteration: {timesteps_per_iter:,}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Target score: {target_score}")
    print(f"  LLM refinement: {'Enabled' if use_llm else 'Disabled'}")
    print()
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}")
        
        # Train for N timesteps
        start_time = time.time()
        model.learn(
            total_timesteps=timesteps_per_iter,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        elapsed = time.time() - start_time
        total_timesteps += timesteps_per_iter
        
        print(f"\nTraining completed in {elapsed:.1f}s")
        print(f"Total timesteps: {total_timesteps:,}")
        
        # Analyze performance
        try:
            analysis = analyzer.analyze(episodes_for_analysis)
            avg_score = analysis.get('avg_score', -21)
            
            print(f"\n--- Performance Analysis ---")
            print(f"Episodes analyzed: {analysis['n_episodes']}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Best Score: {analysis.get('best_score', -21):.0f}")
            print(f"Win Rate: {analysis.get('win_rate', 0):.1f}%")
            print(f"Hit Rate: {analysis.get('hit_rate', 0):.1f}%")
            print(f"Avg Paddle Distance: {analysis.get('avg_paddle_distance', 50):.1f}px")
            print(f"\nBehaviors:\n{analysis.get('behaviors', 'N/A')}")
        except Exception as e:
            print(f"\nERROR in Performance Analysis: {e}")
            print("Skipping analysis for this iteration.")
            avg_score = -21
            analysis = {}
        
        # Update best score
        if avg_score > best_score:
            best_score = avg_score
            # Save best model
            model_path = MODEL_DIR / f"pong_best_iter{iteration}"
            model.save(str(model_path))
            print(f"\n>>> New best score! Model saved to {model_path}")
        
        # Check if target reached
        if avg_score >= target_score:
            print(f"\n*** TARGET SCORE REACHED: {avg_score:.2f} >= {target_score} ***")
            break
        
        # LLM refinement
        if use_llm and iteration < max_iterations:
            print(f"\n--- LLM Reward Refinement ---")
            
            # Get current reward code
            current_code = callback.reward_file.read_text() if callback.reward_file.exists() else None
            
            # Get training summary
            try:
                summary = analyzer.get_summary_for_llm(episodes_for_analysis)
            except Exception as e:
                print(f"Error generating summary for LLM: {e}")
                summary = "Error generating summary."
            
            try:
                # Generate new reward
                new_code, reasoning = llm.generate_reward(summary, current_code)
                
                print(f"LLM Reasoning: {reasoning[:200]}...")
                
                # Save new reward
                llm.save_current_reward(new_code)
                
                # Hot-swap reward function
                callback.reload_reward()
                
                print("Reward function updated!")
                
            except Exception as e:
                print(f"LLM refinement failed: {e}")
                print("Continuing with current reward function...")
        
        # Clear analyzed episodes
        analyzer.clear()
    
    # Training complete
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Best average score: {best_score:.2f}")
    print(f"Iterations: {iteration}")
    
    # Save final model
    final_path = MODEL_DIR / "pong_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")
    
    env.close()
    return model, best_score


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False):
    """Evaluate a trained model"""
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create single env for evaluation
    env = make_env()
    if not render:
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=FRAME_STACK)
        env = VecTransposeImage(env)
    
    scores = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_array, info = env.step(action)
            episode_reward += reward[0]
            done = done_array[0]
            
            if render:
                env.render()
        
        scores.append(episode_reward)
        print(f"Episode {ep+1}: Score = {episode_reward:.0f}")
    
    print(f"\nAverage Score: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Pong with Iterative Reward Refinement")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--timesteps', type=int, default=TIMESTEPS_PER_ITERATION,
                             help='Timesteps per iteration')
    train_parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS,
                             help='Max refinement iterations')
    train_parser.add_argument('--target', type=float, default=TARGET_SCORE,
                             help='Target score to achieve')
    train_parser.add_argument('--no-llm', action='store_true',
                             help='Disable LLM refinement')
    train_parser.add_argument('--verbose', type=int, default=1,
                             help='Verbosity level (0-2)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('model', type=str, help='Path to model file')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of episodes to evaluate')
    eval_parser.add_argument('--render', action='store_true',
                            help='Render the game')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_iterative(
            timesteps_per_iter=args.timesteps,
            max_iterations=args.iterations,
            target_score=args.target,
            use_llm=not args.no_llm,
            verbose=args.verbose,
        )
    elif args.command == 'eval':
        evaluate(args.model, args.episodes, args.render)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
