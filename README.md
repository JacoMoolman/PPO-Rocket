# Moon Lander AI Project - PPO

This project implements a Moon Lander game where an AI agent learns to land a rocket on a target using Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm.

## Project Structure

The project consists of three main components:

1. `GetMoonGame_new.py`: The game environment implementation
2. `moon_ppo.py`: The PPO training implementation
3. `play_trained_moon_lander.py`: Script to watch the trained agent play

## Requirements

```
pygame
gymnasium
stable-baselines3
torch
numpy
```

## Game Environment (`GetMoonGame_new.py`)

The Moon Lander game is implemented using Pygame. The rocket must be safely landed on a target location while managing:

- Gravity
- Thrust
- Rotation
- Fuel consumption
- Landing precision

Key features:
- Realistic physics simulation with gravity and momentum
- Visual feedback with rocket flames during thrust
- Dynamic target placement
- Score tracking and performance metrics

Game Parameters:
- Gravity: 0.05 units/frame²
- Thrust: 0.5 units/frame
- Rotation Speed: 3 degrees/frame
- Max Speed: 3 units/frame
- Max Angular Velocity: 10 degrees/frame

## Training Implementation (`moon_ppo.py`)

This file implements the reinforcement learning environment and training logic using PPO algorithm.

### State Space (8 dimensions):
1. Rocket X position (normalized)
2. Rocket Y position (normalized)
3. Rocket angle (normalized)
4. X velocity (normalized)
5. Y velocity (normalized)
6. Distance to target (normalized)
7. X distance to target (normalized)
8. Y distance to target (normalized)

### Action Space (3 dimensions):
1. Right rotation (continuous: -1 to 1)
2. Left rotation (continuous: -1 to 1)
3. Thrust (continuous: -1 to 1)

### Neural Network Architecture:
- Policy Network: 10 layers of 512 units each
- Value Network: 10 layers of 512 units each

### Training Parameters:
- Learning Rate: 5e-4
- Batch Size: 256
- Steps per Update: 2048
- Training Epochs: 10
- Gamma (discount factor): 0.99
- GAE Lambda: 0.95
- Clip Range: 0.2
- Entropy Coefficient: 0.01

The training process uses multiple parallel environments (default: 10) to gather diverse experiences and improve training stability.

## Playing the Trained Agent (`play_trained_moon_lander.py`)

This script loads the latest trained model and demonstrates its performance in the game environment.

Features:
- Automatically finds and loads the latest checkpoint
- Runs at 60 FPS for smooth visualization
- Displays real-time performance metrics
- Allows multiple episodes to be played consecutively

## How to Use

1. **Training a New Agent:**
   ```bash
   python moon_ppo.py
   ```
   - Training checkpoints are saved every 10,000 steps
   - Training can be interrupted safely with Ctrl+C
   - Progress is logged to text files for each environment

2. **Playing with Trained Agent:**
   ```bash
   python play_trained_moon_lander.py
   ```
   - Automatically loads the most recent checkpoint
   - Close the window to end the demonstration

## Performance Metrics

The agent is evaluated on:
- Landing precision (distance to target)
- Landing speed (must be below threshold)
- Fuel efficiency (thrust usage)
- Landing angle (must be close to vertical)
- Total episode reward

## Tips for Training

1. Training typically requires several million steps for good performance
2. Monitor the reward logs to track learning progress
3. Adjust hyperparameters if the agent is not learning effectively:
   - Increase network size for more complex behavior
   - Adjust learning rate if training is unstable
   - Modify reward structure in the environment if needed

## Acknowledgments

This implementation uses:
- PyTorch for neural networks
- Stable-Baselines3 for PPO implementation
- Pygame for visualization
- Gymnasium for the RL environment structure
