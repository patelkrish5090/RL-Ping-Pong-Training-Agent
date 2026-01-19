"""
Script to visualize the trained Pong agent playing.
"""
import sys
import time
from pathlib import Path

# Add parent directory to path to import config/utils
import shimmy
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import config (ensure we're matching training settings)
from config import ENV_ID, FRAME_STACK

def main():
    model_path = PROJECT_ROOT / "models" / "pong_final"
    
    if not model_path.exists():
        # Fallback to checking for zip extension if needed
        model_path = model_path.with_suffix(".zip")
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return

    print(f"Loading model from {model_path}...")
    # Load model (force CPU for inference compatibility)
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Creating environment...")
    # Create env with human render mode
    env = gym.make(ENV_ID, render_mode="human")
    env = AtariWrapper(env)
    
    # Needs to match the training wrapping for frame stacking
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)

    print("\nStarting gameplay loop...")
    print("---------------------------------------------------------")
    print("  RIGHT PREDDLE (Green)  = Your Trained AI Agent")
    print("  LEFT PADDLE (Orange)   = Built-in CPU Opponent")
    print("---------------------------------------------------------")
    print("Press Ctrl+C in the terminal to stop.")
    
    try:
        episodes = 0
        while True:
            obs = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                
                # Step env (render handled by render_mode="human")
                obs, reward, done_array, info = env.step(action)
                
                total_reward += reward[0]
                done = done_array[0]
                
                # Optional: Add small sleep if it's too fast (though Atari usually syncs)
                # time.sleep(0.01) 
            
            episodes += 1
            print(f"Episode {episodes}: Score = {total_reward:.0f}")
            
            # Short pause between episodes
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping gameplay.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
