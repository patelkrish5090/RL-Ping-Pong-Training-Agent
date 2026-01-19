# Complete Research Context: Optimizing Reinforcement Learning with LLMs

> **Purpose**: This document contains ALL research, approaches, implementation plans, and context for the research paper on optimizing RL with LLMs. If starting a new chat session, read this file first to understand the complete project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Question](#research-question)
3. [Background Knowledge](#background-knowledge)
4. [Complete Literature Review](#complete-literature-review)
5. [All Approaches Explained](#all-approaches-explained)
6. [User's Primary Approach](#users-primary-approach)
7. [Implementation Plans for Each Approach](#implementation-plans-for-each-approach)
8. [Experimental Design](#experimental-design)
9. [Technical Setup](#technical-setup)
10. [Paper Structure](#paper-structure)
11. [Key Findings from Research](#key-findings-from-research)
12. [Open Questions](#open-questions)
13. [Timeline](#timeline)
14. [References](#references)

---

## Project Overview

**Goal**: Write a research paper demonstrating how Large Language Models (LLMs) can optimize Reinforcement Learning (RL) training for games.

**User's Setup**:
- Has local LLM (Qwen) available
- Has GPU available
- No time or compute constraints
- Wants to test on games (Pong, etc.)

**Target Outcome**: A research paper comparing different LLM-assisted RL optimization approaches, with the user's iterative refinement approach as the main contribution.

---

## Research Question

> **Primary Question**: "How can LLMs help make Reinforcement Learning faster and more efficient for game agents?"

**Sub-questions**:
1. Can LLMs generate effective reward functions for games?
2. Is iterative refinement (updating rewards during training) better than one-shot generation?
3. Which games benefit most from LLM assistance?
4. What is the optimal frequency for LLM feedback (every N episodes)?
5. How does LLM-based reward design compare to curriculum learning?

---

## Background Knowledge

### What is Reinforcement Learning (RL)?

RL is a type of machine learning where an **agent** learns by interacting with an **environment**:

```
Agent takes ACTION → Environment gives STATE + REWARD → Agent learns
```

**Key Components**:
- **Agent**: The learner (e.g., game-playing AI)
- **Environment**: The game/simulation
- **State**: Current situation (e.g., ball position, paddle position)
- **Action**: What agent can do (e.g., move up, move down)
- **Reward**: Feedback signal (+1 for scoring, -1 for losing)
- **Policy**: Agent's strategy (maps states to actions)

### The Sparse Reward Problem

Most games only give rewards rarely:
- Pong: +1 when you score, -1 when opponent scores (nothing in between)
- This makes learning slow because agent doesn't know if it's improving

### What is Reward Shaping?

Adding intermediate rewards to guide learning:
- Original: +1 for scoring
- Shaped: +0.1 for moving paddle toward ball, +0.2 for hitting ball, +1 for scoring

**Challenge**: Designing good shaped rewards is difficult and time-consuming.

### What are LLMs?

Large Language Models (GPT-4, Claude, Qwen) are AI models trained on text that can:
- Understand natural language descriptions
- Write code
- Reason about problems
- Generate solutions

**Key Insight**: LLMs can understand game objectives and write reward function code!

### What is PPO?

Proximal Policy Optimization (PPO) is a popular RL algorithm:
- Stable and reliable
- Works well for games
- Used as the standard algorithm in this research

---

## Complete Literature Review

### 1. Eureka (NVIDIA, ICLR 2024)

**Paper**: "Eureka: Human-Level Reward Design via Coding Large Language Models"

**What it does**:
- Uses GPT-4 to generate reward function code
- Evolutionary search: generates multiple rewards, keeps best, iterates
- Tested on 29 robotics tasks in IsaacGym

**Key Results**:
- 83% of tasks: Eureka beats human-designed rewards
- 52% average improvement over human baselines
- Famous demo: Robot hand spinning a pen

**How Eureka Works**:
```
1. Give GPT-4 the environment source code
2. GPT-4 generates reward function (Python code)
3. Train RL agent with that reward
4. Send training statistics back to GPT-4
5. GPT-4 generates improved reward
6. Repeat 5 iterations with 16 samples each
```

**Limitations Discovered**:
- Only tested on robotics (IsaacGym), NOT games
- Only works with state-based observations, NOT pixels
- Expensive: ~80 GPT-4 calls per task
- Reproducibility issues: Isaac Gym deprecated, dependencies broken
- Claims are based on averages - hides failures
- No real-world robot testing

**Compute Cost** (estimated):
- LLM API: $24-48 per task
- GPU training: $230-400 per full experiment
- Total: Very expensive for full replication

**What This Means for Our Research**:
- Eureka proves the concept works
- But we need to test on GAMES, not just robots
- We need simpler, more reproducible approach

---

### 2. Text2Reward (ICLR 2024 Spotlight)

**Paper**: "Text2Reward: Reward Shaping with Language Models for Reinforcement Learning"

**What it does**:
- LLM generates reward functions from natural language goals
- Adds human feedback loop for refinement
- Three stages: Expert Abstraction → User Instruction → User Feedback

**Key Results**:
- 13/17 manipulation tasks: matched or beat human rewards
- 6 locomotion behaviors learned with 94% success
- Successfully deployed on real robots

**How Text2Reward Works**:
```
1. Human describes goal in natural language
2. LLM generates reward code
3. Train agent
4. Human observes behavior, gives feedback
   "The robot is dropping objects too often"
5. LLM refines reward based on feedback
6. Repeat until satisfactory
```

**Difference from Eureka**:
- Text2Reward: Human-in-the-loop feedback
- Eureka: Automatic feedback from training statistics

**What This Means for Our Research**:
- Human feedback can help LLM improve rewards
- But we want AUTOMATIC feedback (like Eureka)
- Our approach: Use training statistics as automatic feedback

---

### 3. RLVR - Reinforcement Learning from Verifiable Rewards (2025)

**Used by**: DeepSeek-R1 (famous reasoning model)

**What it does**:
- Instead of complex rewards, use simple verifiable rewards
- For math: Did you get the right answer? Yes = +1, No = 0
- For code: Does it run correctly? Yes = +1, No = 0

**Key Insight**:
- Sometimes simple binary rewards work better than complex shaped rewards
- If you can verify correctness automatically, you don't need reward engineering

**How RLVR Works**:
```
Task: Solve 2 + 2 = ?
Agent answers: 4
Reward: +1 (correct, verifiable)

Task: Solve 2 + 2 = ?
Agent answers: 5  
Reward: 0 (wrong)
```

**What This Means for Our Research**:
- For games with clear win conditions, RLVR might be simpler
- But many games need intermediate feedback
- We focus on cases where reward shaping IS helpful

---

### 4. Cooper (August 2025)

**Paper**: "Co-optimizing Policy and Reward Models"

**Problem it Solves**:
- Reward hacking: Agent exploits bugs in reward function
- Example: Agent finds loophole to get high reward without solving task

**What Cooper Does**:
- Trains policy AND reward model together
- If agent finds exploit, reward model updates to fix it
- Co-evolution prevents gaming the system

**How Cooper Works**:
```
1. Initial reward model + policy
2. Train policy
3. If policy exploits reward → Update reward model
4. Continue training
5. Both improve together
```

**What This Means for Our Research**:
- Reward hacking is a real problem
- Our iterative approach might naturally avoid this
- We update rewards based on BEHAVIOR, not just score

---

### 5. SPARK (September 2025)

**Paper**: "Synergistic Policy And Reward co-evolving"

**Similar to Cooper**: Joint evolution of policy and reward

**Key Insight**: Static rewards become outdated as agent improves

**What This Means for Our Research**:
- Supports our iterative refinement idea
- Rewards should adapt as agent learns

---

### 6. Text2Grad (Microsoft, May 2025)

**Paper**: "Converting Natural Language Feedback into Span-Level Gradients"

**What it does**:
- Instead of scalar rewards, gives detailed per-token feedback
- "Word 3 was good (+0.8), Word 5 was bad (-0.2)"

**Very Precise Feedback**:
```
Normal reward: "Your response scored 7/10"
Text2Grad: "Token 1: +0.9, Token 2: +0.3, Token 3: -0.5, ..."
```

**Best For**: Text generation tasks (summarization, code)

**What This Means for Our Research**:
- Shows importance of detailed feedback
- For games, we give behavioral feedback instead of token-level
- Different domain but same principle: more feedback = better learning

---

### 7. CARD (Tsinghua University, 2025)

**Paper**: "Dynamic Reward Adaptation Framework"

**What it does**:
- LLM-driven reward design with dynamic feedback
- Updates rewards without human intervention
- Uses text-code reconciliation

**Key Claim**: Lower human cost, token usage, and training time than Eureka

**What This Means for Our Research**:
- Validates automatic feedback approach
- Our work is similar but focused on games specifically

---

### 8. R* (ICML 2025)

**Paper**: "Reward Structure Evolution"

**What it does**:
- Decomposes reward into structure + parameters
- LLMs mutate reward structure
- Optimization tunes parameters

**Key Insight**: Separating structure from parameters is more efficient

**What This Means for Our Research**:
- Could try this as an advanced variant
- But adds complexity - start simple first

---

### 9. NitroGen (NVIDIA + Stanford, December 2025)

**Paper**: "An Open Foundation Model for Generalist Gaming Agents"

**What it does**:
- Learns to play games by watching YouTube gameplay
- 40,000 hours of gameplay across 1,000+ games
- Uses behavior cloning (imitation), NOT RL

**Important Note**:
- This is NOT RL optimization
- It's imitation learning (copying humans)
- Different approach entirely

**What This Means for Our Research**:
- Shows NVIDIA is investing in game AI
- But NOT relevant to our RL focus
- We optimize RL training, not replace it with imitation

---

### 10. NVIDIA ACE (2025)

**What it does**:
- Makes NPCs (game characters) smarter
- Uses small language models (SLMs) for NPC decisions
- Real games: PUBG, Naraka Bladepoint

**What This Means for Our Research**:
- Shows LLMs in games are a hot topic
- But ACE is for NPCs, not training agents
- Not directly related to our work

---

## All Approaches Explained

### Overview: 6 Ways to Use LLMs for RL Optimization

```
┌─────────────────────────────────────────────────────────────────────────┐
│              6 APPROACHES TO OPTIMIZE RL WITH LLMs                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  APPROACH 1: ITERATIVE REWARD REFINEMENT ← User's Main Approach         │
│  └─ LLM updates reward every N episodes based on training progress      │
│                                                                         │
│  APPROACH 2: ONE-SHOT REWARD GENERATION                                 │
│  └─ LLM generates reward once, no updates                               │
│                                                                         │
│  APPROACH 3: EVOLUTIONARY SEARCH (Eureka-style)                         │
│  └─ Generate multiple rewards, evaluate, keep best, evolve              │
│                                                                         │
│  APPROACH 4: CURRICULUM LEARNING                                        │
│  └─ LLM designs difficulty progression (easy → hard)                    │
│                                                                         │
│  APPROACH 5: ACTION ADVISING                                            │
│  └─ LLM suggests actions during training                                │
│                                                                         │
│  APPROACH 6: HYBRID (Iterative + Curriculum)                            │
│  └─ Combine reward refinement with curriculum                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Approach 1: Iterative Reward Refinement (USER'S PRIMARY APPROACH)

**Core Idea**: Update the reward function DURING training based on agent's progress.

**How It Works**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ITERATIVE REWARD REFINEMENT                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   START: Initial reward function (from LLM or simple default)          │
│                          ↓                                              │
│   LOOP:                                                                 │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  1. Train for N episodes (e.g., 10)                          │     │
│   │                          ↓                                    │     │
│   │  2. Collect statistics:                                       │     │
│   │     - Average score                                           │     │
│   │     - Episode lengths                                         │     │
│   │     - Common failure modes                                    │     │
│   │     - Behavior patterns                                       │     │
│   │                          ↓                                    │     │
│   │  3. Send to LLM with prompt:                                  │     │
│   │     "Here's current reward and results. Improve it."          │     │
│   │                          ↓                                    │     │
│   │  4. LLM generates updated reward function                     │     │
│   │                          ↓                                    │     │
│   │  5. Replace reward, continue training                         │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                          ↓                                              │
│   END: When target performance reached or max iterations                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why This Is Novel**:
- Eureka: Updates between FULL training runs (expensive)
- Our approach: Updates DURING training (faster adaptation)
- More sample efficient
- LLM sees real behavioral feedback

**Key Parameters**:
- N (episodes between updates): 10, 50, or 100?
- What info to send: scores only? trajectories? failure analysis?
- How to describe behaviors to LLM

**Example for Pong**:
```
Episode 1-10: Score = -15 (losing badly)
→ LLM: "Agent doesn't track ball. Add reward for paddle-ball alignment."

Episode 11-20: Score = -8 (improving)
→ LLM: "Agent tracks but misses. Add reward for being in ball's path."

Episode 21-30: Score = +5 (winning sometimes)
→ LLM: "Good progress. Add reward for hitting ball toward corners."

... continue until score > +18 ...
```

---

### Approach 2: One-Shot Reward Generation

**Core Idea**: LLM generates reward once, no iteration.

**How It Works**:
```
1. Describe game to LLM
2. LLM generates reward function
3. Train fully with that reward
4. Done
```

**Pros**:
- Simple
- Cheap (one LLM call)
- Fast to implement

**Cons**:
- No adaptation
- If reward is bad, training fails
- Misses opportunity to improve

**Purpose in Our Research**:
- Baseline to compare against iterative approach
- Show that iteration IS valuable

---

### Approach 3: Evolutionary Search (Eureka-style)

**Core Idea**: Generate multiple rewards, evaluate each, keep best, evolve.

**How It Works**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY SEARCH                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Iteration 1:                                                          │
│   ├─ LLM generates Reward A, B, C, D (4 candidates)                     │
│   ├─ Train each for 10K steps (short evaluation)                        │
│   ├─ Score: A=50, B=120, C=80, D=30                                     │
│   └─ Best: B (keep this)                                                │
│                          ↓                                              │
│   Iteration 2:                                                          │
│   ├─ Tell LLM: "B worked best because... Generate variants"             │
│   ├─ LLM generates B1, B2, B3, B4 (variants of B)                       │
│   ├─ Train each for 10K steps                                           │
│   ├─ Score: B1=130, B2=140, B3=110, B4=125                              │
│   └─ Best: B2                                                           │
│                          ↓                                              │
│   ... repeat for 5 iterations ...                                       │
│                          ↓                                              │
│   Final: Use best reward for full training                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pros**:
- Explores diverse reward designs
- Selection pressure finds good solutions
- Theoretically finds better optima

**Cons**:
- Expensive (many LLM calls + many training runs)
- Slow
- Requires parallel compute for efficiency

**Difference from Our Approach**:
- Eureka: Parallel search, pick best at end
- Ours: Sequential refinement during single training run
- Ours is more sample efficient

---

### Approach 4: Curriculum Learning

**Core Idea**: LLM designs difficulty progression, not rewards.

**How It Works**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CURRICULUM LEARNING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   LLM designs curriculum:                                               │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Level 1 (Easy):   Slow ball, big paddle, no wind               │   │
│   │ Level 2 (Medium): Normal ball, normal paddle, no wind          │   │
│   │ Level 3 (Hard):   Fast ball, normal paddle, wind               │   │
│   │ Level 4 (Expert): Fast ball, small paddle, strong wind         │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                          ↓                                              │
│   Training:                                                             │
│   ├─ Train on Level 1 until solved                                      │
│   ├─ Move to Level 2, continue training                                 │
│   ├─ Move to Level 3...                                                 │
│   └─ Graduate to Level 4 (full difficulty)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pros**:
- Doesn't modify reward function
- Progressive difficulty helps learning
- Few LLM calls needed

**Cons**:
- Requires configurable difficulty
- Not all games support this
- Doesn't address sparse reward directly

**When to Use**:
- Games with adjustable parameters
- When reward shaping is risky (might cause reward hacking)

---

### Approach 5: Action Advising

**Core Idea**: LLM suggests actions during training.

**How It Works**:
```
During training:
  Agent: "I'm in state [x, y, velocity]. What should I do?"
  LLM: "Fire the main engine to slow down."
  Agent: Uses this advice with some probability
```

**Pros**:
- Direct guidance
- Can help with exploration

**Cons**:
- LLM is SLOW (not real-time)
- Many LLM calls (every step?)
- Expensive

**Practical Limitation**:
- Only works for slow games
- Not suitable for Pong (too fast)
- Better for turn-based games

**What This Means for Our Research**:
- Probably not practical for our games
- Mention as alternative but don't implement

---

### Approach 6: Hybrid (Iterative + Curriculum)

**Core Idea**: Combine reward refinement with curriculum.

**How It Works**:
```
1. LLM designs curriculum (easy → hard)
2. Start training on easy level
3. Every N episodes: LLM updates reward based on progress
4. When easy is solved: Move to next difficulty
5. Continue iterative refinement at each level
```

**Pros**:
- Best of both approaches
- Maximum LLM assistance

**Cons**:
- Most complex to implement
- More LLM calls

**When to Use**:
- If single approach isn't enough
- For very difficult games

---

## Users Primary Approach

### Detailed Specification

**Approach Name**: Iterative Reward Refinement with LLM Feedback

**Core Loop**:
```python
# Pseudocode for user's approach

def train_with_iterative_refinement(env, llm, N=10, max_episodes=1000):
    # Initial reward (simple or LLM-generated)
    reward_fn = llm.generate_initial_reward(env.description)
    
    episode = 0
    while episode < max_episodes:
        # Train for N episodes
        stats = train_n_episodes(env, reward_fn, N)
        episode += N
        
        # Check if done
        if stats.avg_score > TARGET_SCORE:
            break
        
        # Ask LLM to improve reward
        feedback = format_feedback(stats)
        reward_fn = llm.refine_reward(reward_fn, feedback)
    
    return trained_agent
```

**Information to Send to LLM**:
```
Current reward function:
[CODE]

Last 10 episodes:
- Average score: -8.5
- Best score: +3
- Worst score: -15
- Average episode length: 245 steps
- Common behaviors:
  * Agent tracks ball 60% of time
  * Agent misses ball when moving fast
  * Agent rarely moves to corners
- Failure modes:
  * 40% of losses: Ball passed while agent moving wrong direction
  * 30% of losses: Ball too fast to reach
  * 30% of losses: Agent stuck in corner

Suggest improvements to the reward function.
```

**Parameters to Experiment With**:

| Parameter | Options | Trade-off |
|-----------|---------|-----------|
| N (update frequency) | 5, 10, 25, 50, 100 | More frequent = more LLM calls, faster adaptation |
| Info detail level | Basic, Medium, Detailed | More detail = better LLM understanding, more tokens |
| LLM temperature | 0.5, 0.7, 1.0 | Higher = more creative, less stable |

---

## Implementation Plans for Each Approach

### Implementation Plan: Approach 1 (Iterative Refinement)

**Step 1: Setup Environment**
```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")
```

**Step 2: Create Reward Wrapper**
```python
class DynamicRewardWrapper(gym.Wrapper):
    def __init__(self, env, initial_reward_fn):
        super().__init__(env)
        self.reward_fn = initial_reward_fn
        self.episode_data = []
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Apply custom reward
        shaped_reward = self.reward_fn(obs, action, reward, done)
        return obs, shaped_reward, done, truncated, info
    
    def update_reward(self, new_reward_fn):
        self.reward_fn = new_reward_fn
```

**Step 3: LLM Integration**
```python
import ollama

def ask_llm_for_reward(current_code, stats):
    prompt = f"""
    Current reward function:
    ```python
    {current_code}
    ```
    
    Training results:
    - Average score: {stats['avg_score']}
    - Common failures: {stats['failures']}
    
    Improve the reward function. Return only Python code.
    """
    
    response = ollama.chat(
        model='qwen2.5-coder:7b-instruct',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return extract_code(response['message']['content'])
```

**Step 4: Training Loop**
```python
def train_iterative(env, n_episodes_per_update=10, max_updates=20):
    wrapper = DynamicRewardWrapper(env, initial_reward)
    model = PPO("MlpPolicy", wrapper)
    
    stats_history = []
    
    for update in range(max_updates):
        # Train for N episodes
        model.learn(total_timesteps=n_episodes_per_update * 500)
        
        # Collect stats
        stats = evaluate(model, env)
        stats_history.append(stats)
        
        # Ask LLM to improve
        new_reward = ask_llm_for_reward(current_reward_code, stats)
        wrapper.update_reward(new_reward)
        
        # Check if solved
        if stats['avg_score'] > TARGET:
            break
    
    return model, stats_history
```

---

### Implementation Plan: Approach 2 (One-Shot)

**Simple - Just one LLM call**:
```python
def train_one_shot(env):
    # Get reward from LLM once
    reward_code = ask_llm_for_initial_reward(env.description)
    reward_fn = compile_reward(reward_code)
    
    # Train fully with that reward
    wrapper = RewardWrapper(env, reward_fn)
    model = PPO("MlpPolicy", wrapper)
    model.learn(total_timesteps=500000)
    
    return model
```

---

### Implementation Plan: Approach 3 (Evolutionary)

```python
def train_evolutionary(env, n_candidates=4, n_generations=5):
    best_reward = None
    best_score = -float('inf')
    
    for gen in range(n_generations):
        # Generate candidates
        if gen == 0:
            candidates = [ask_llm_for_initial_reward() for _ in range(n_candidates)]
        else:
            candidates = [ask_llm_for_variant(best_reward) for _ in range(n_candidates)]
        
        # Evaluate each
        scores = []
        for reward_code in candidates:
            score = evaluate_reward(env, reward_code, timesteps=50000)
            scores.append(score)
        
        # Keep best
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_reward = candidates[best_idx]
    
    # Final training with best reward
    return train_with_reward(env, best_reward, timesteps=500000)
```

---

### Implementation Plan: Approach 4 (Curriculum)

```python
def train_curriculum(env_class):
    # Get curriculum from LLM
    curriculum = ask_llm_for_curriculum(env_class.description)
    # Example: [{'gravity': -5}, {'gravity': -10}, {'gravity': -10, 'wind': 10}]
    
    model = None
    for level, params in enumerate(curriculum):
        env = env_class(**params)
        
        if model is None:
            model = PPO("MlpPolicy", env)
        else:
            model.set_env(env)
        
        # Train until solved at this level
        while not solved(model, env):
            model.learn(total_timesteps=10000)
    
    return model
```

---

## Experimental Design

### Games to Test

| Game | Type | Difficulty | State Size | Actions | Why Include |
|------|------|------------|------------|---------|-------------|
| CartPole-v1 | Balance | Easy | 4 | 2 | Baseline - LLM may not help |
| LunarLander-v2 | Control | Medium | 8 | 4 | Good test case |
| MountainCar-v0 | Exploration | Hard | 2 | 3 | LLM should help most here |
| Pong (Atari) | Game | Medium | Pixels/RAM | 6 | User's original target |

### Metrics to Measure

1. **Training Efficiency**
   - Episodes to reach target score
   - Total timesteps to solve
   - Wall-clock time

2. **Final Performance**
   - Best score achieved
   - Average score (last 100 episodes)
   - Score variance

3. **LLM Cost**
   - Number of LLM calls
   - Total tokens used
   - Time spent on LLM inference

4. **Learning Quality**
   - Learning curve smoothness
   - Stability (variance between runs)

### Experimental Conditions

| Condition | Description | LLM Calls |
|-----------|-------------|-----------|
| Baseline | PPO with default reward | 0 |
| One-Shot | LLM generates reward once | 1 |
| Iterative-10 | Update every 10 episodes | Many |
| Iterative-50 | Update every 50 episodes | Medium |
| Iterative-100 | Update every 100 episodes | Fewer |
| Curriculum | LLM designs difficulty | 1 |
| Hybrid | Curriculum + Iterative | Medium |

### Expected Results (Hypothesis)

| Game | Baseline | One-Shot | Iterative | Curriculum |
|------|----------|----------|-----------|------------|
| CartPole | 50K | 48K | 45K | 40K |
| LunarLander | 300K | 250K | 180K | 200K |
| MountainCar | 1M+ | 500K | 300K | 350K |

**Key Hypothesis**: Iterative approach helps MOST on hard exploration games.

---

## Technical Setup

### Software Requirements

```bash
# Python 3.10+
pip install stable-baselines3[extra]
pip install gymnasium[classic-control]
pip install gymnasium[box2d]
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install ollama
pip install numpy pandas matplotlib
```

### Hardware

- GPU: Available (user has no constraints)
- CPU: Any modern CPU
- RAM: 16GB+ recommended

### LLM Setup

Using local Qwen via Ollama:
```bash
# Install Ollama
# Windows: Download from https://ollama.com/download/windows

# Pull model
ollama pull qwen2.5-coder:7b-instruct

# Test
ollama run qwen2.5-coder:7b-instruct "Write a hello world in Python"
```

### Key Hyperparameters

**PPO Settings**:
```python
ppo_config = {
    'learning_rate': 2.5e-4,
    'n_steps': 128,
    'batch_size': 256,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'clip_range': 0.1,
}
```

**LLM Settings**:
```python
llm_config = {
    'model': 'qwen2.5-coder:7b-instruct',
    'temperature': 1.0,  # High for creativity
    'num_ctx': 8192,     # Context window
}
```

---

## Paper Structure

### Title Options

1. "Iterative Reward Refinement: Using LLMs to Optimize Reinforcement Learning"
2. "Dynamic Reward Shaping with Large Language Models for Game AI"
3. "When Do LLMs Help? A Study of LLM-Assisted RL Optimization"

### Detailed Outline

**1. Introduction (1-2 pages)**
- RL is powerful but training is slow
- Reward design is critical but difficult
- LLMs can understand tasks and write code
- Our contribution: Iterative reward refinement
- Preview of results

**2. Background (2-3 pages)**
- 2.1 Reinforcement Learning Basics
  - Agent, environment, reward
  - Policy optimization
  - Sparse vs dense rewards
- 2.2 The Reward Design Problem
  - Why reward shaping helps
  - Why it's hard to design
- 2.3 Large Language Models
  - What they are
  - Code generation capability
- 2.4 Related Work
  - Eureka
  - Text2Reward
  - RLVR
  - Cooper/SPARK

**3. Methods (3-4 pages)**
- 3.1 Problem Formulation
  - Formal RL setup
  - Goal: minimize training time
- 3.2 Our Approach: Iterative Refinement
  - Algorithm description
  - Information sent to LLM
  - Reward update process
- 3.3 Comparison Approaches
  - One-shot generation
  - Evolutionary search
  - Curriculum learning
- 3.4 Implementation Details
  - Games used
  - LLM configuration
  - PPO settings

**4. Experiments (4-5 pages)**
- 4.1 Experimental Setup
  - Games and metrics
  - Baselines
  - Evaluation protocol
- 4.2 Main Results
  - Comparison table
  - Learning curves
  - Statistical analysis
- 4.3 Analysis
  - When does LLM help?
  - Effect of update frequency (N)
  - Quality of generated rewards
- 4.4 Ablation Studies
  - Different info levels
  - Different LLM temperatures

**5. Discussion (1-2 pages)**
- 5.1 Key Findings
  - Iterative > One-shot
  - LLM helps more on hard games
  - Optimal update frequency
- 5.2 Limitations
  - LLM quality matters
  - Compute cost
  - Game-specific tuning
- 5.3 Future Work
  - Multi-game generalization
  - Automatic behavior analysis
  - Hybrid approaches

**6. Conclusion (1 page)**
- Summary of contributions
- Practical recommendations
- Impact on field

---

## Key Findings from Research

### What the Literature Says

1. **LLM reward design DOES work** for robotics (Eureka proved this)

2. **But hasn't been tested on games** - this is our contribution

3. **Simple games (Pong, CartPole) may not benefit** - they're already easy

4. **Hard exploration games should benefit most** - this is where reward shaping helps

5. **Iterative is probably better than one-shot** - Cooper/SPARK support this

6. **Reward hacking is a risk** - need to monitor for exploits

7. **Local LLMs (Qwen) should work** - but quality may be lower than GPT-4

### What We Need to Prove

1. LLM reward design works for GAMES (not just robots)
2. Iterative refinement is better than one-shot
3. Quantify the speedup (% faster training)
4. Identify when it helps vs. when it doesn't

---

## Open Questions

1. **What's the optimal update frequency (N)?**
   - Need to experiment: 10, 25, 50, 100

2. **What information should we send to LLM?**
   - Minimal: Just scores
   - Medium: Scores + episode lengths + failure counts
   - Detailed: Full behavior analysis

3. **How do we detect and prevent reward hacking?**
   - Monitor for unexpected behaviors
   - Validate reward before using

4. **Can we use this for pixel-based games (Atari)?**
   - Need different approach for image observations
   - Maybe describe game state in words

5. **How does LLM quality affect results?**
   - Compare Qwen 7B vs larger models (if possible)

---

## Timeline

| Week | Tasks |
|------|-------|
| 1 | Setup environment, implement baseline PPO |
| 2 | Implement one-shot and iterative approaches |
| 3 | Run experiments on CartPole, LunarLander |
| 4 | Run experiments on MountainCar, (optionally Pong) |
| 5 | Analyze results, create visualizations |
| 6 | Write paper |

---

## References

### Core Papers

1. Ma et al. "Eureka: Human-Level Reward Design via Coding Large Language Models" ICLR 2024
   - https://arxiv.org/abs/2310.12931

2. "Text2Reward: Reward Shaping with Language Models for Reinforcement Learning" ICLR 2024
   - https://arxiv.org/abs/2309.11489

3. Schulman et al. "Proximal Policy Optimization Algorithms" 2017
   - https://arxiv.org/abs/1707.06347

### Related Work 2025

4. Cooper: "Co-optimizing Policy and Reward Models" 2025
   - https://arxiv.org/abs/2508.05613

5. SPARK: "Synergistic Policy And Reward co-evolving" 2025
   - https://arxiv.org/abs/2509.22624

6. Text2Grad: Microsoft, 2025
   - https://arxiv.org/abs/2505.22338

7. CARD: Dynamic Reward Framework, 2025
   - https://arxiv.org/abs/2504.07596

### Tools

8. Stable-Baselines3: https://stable-baselines3.readthedocs.io/
9. Gymnasium: https://gymnasium.farama.org/
10. Ollama: https://ollama.com/

### NVIDIA (Not directly related but mentioned)

11. NitroGen: Gaming foundation model (behavior cloning, not RL)
    - https://arxiv.org/abs/2601.02427

12. NVIDIA ACE: Autonomous NPCs
    - https://developer.nvidia.com/ace

---

## Quick Reference for New Chat

**If starting a new chat, here's the key context:**

```
PROJECT: Research paper on optimizing RL with LLMs

USER'S MAIN APPROACH: 
- Iterative reward refinement
- Train for N episodes → Send stats to LLM → LLM updates reward → Repeat
- Focus on games (Pong, LunarLander, etc.)

COMPARISON APPROACHES:
- Baseline (no LLM)
- One-shot (LLM generates reward once)
- Curriculum (LLM designs difficulty progression)

USER HAS:
- Local Qwen LLM
- GPU available
- No time/compute constraints

NEXT STEPS:
1. Finalize experimental parameters
2. Implement baseline PPO
3. Implement iterative refinement
4. Run experiments
5. Write paper

KEY INSIGHT FROM RESEARCH:
- Eureka works for robots but not tested on games
- Iterative approaches (Cooper, SPARK) are state-of-the-art 2025
- Simple games may not benefit, hard games should
```

---

## Changelog

- **January 2026**: Initial research and planning complete
- **TODO**: Implementation phase
- **TODO**: Experiments
- **TODO**: Paper writing

---

*End of Complete Context Document*
