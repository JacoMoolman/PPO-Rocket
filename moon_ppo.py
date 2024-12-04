import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import pygame
from GetMoonGame_new import MoonLanderGame
import os

# Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class MoonLanderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Create a dummy network for MoonLanderGame
        class DummyNetwork:
            def activate(self, x):
                return [0, 0, 0]
        
        self.game = MoonLanderGame(DummyNetwork())
        
    def step(self, action):
        # Convert continuous actions to discrete with more granularity
        discrete_action = [
            float(action[0] > 0.5),  # Strong right rotation
            float(action[0] > 0.1 and action[0] <= 0.5),  # Weak right rotation
            float(action[1] > 0.5),  # Strong left rotation
            float(action[1] > 0.1 and action[1] <= 0.5),  # Weak left rotation
            float(action[2] > 0.7),  # Strong thrust
            float(action[2] > 0.3 and action[2] <= 0.7),  # Medium thrust
            float(action[2] > 0 and action[2] <= 0.3)  # Weak thrust
        ]
        
        # Override the network's activate function
        self.game.net.activate = lambda x: discrete_action
        
        # Run one step of the game
        state, reward, done = self.game.run_step()
        
        if state is None:  # Game quit
            return np.zeros(8), 0, True, True, {}
            
        return np.array(state, dtype=np.float32), reward, done, False, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game.reset_game()
        state = [
            self.game.position.x / self.game.WIDTH,
            self.game.position.y / self.game.HEIGHT,
            self.game.angle / 360,
            self.game.velocity.x / self.game.MAX_SPEED,
            self.game.velocity.y / self.game.MAX_SPEED,
            self.game.position.distance_to(self.game.target_pos) / ((self.game.WIDTH**2 + self.game.HEIGHT**2)**0.5),
            (self.game.target_pos.x - self.game.position.x) / self.game.WIDTH,
            (self.game.target_pos.y - self.game.position.y) / self.game.HEIGHT,
        ]
        return np.array(state, dtype=np.float32), {}
    
    def render(self):
        # Rendering is handled by the game class
        pass
    
    def close(self):
        pygame.quit()

def make_env():
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = MoonLanderEnv()
        # Wrap environment with Monitor for logging
        env = Monitor(env)
        return env
    return _init

def train_ppo(num_envs=10):  
    # Create multiple environments for parallel training
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Create the PPO model with custom parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,  # Decreased from 1e-3 to 5e-4 for more gradual learning
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,  # Increased from 0.02 to 0.1 to force more exploration
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512],  # 10 layers for policy
                vf=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512]   # 10 layers for value
            )
        )
    )
    
    try:
        # Train the agent
        model.learn(
            total_timesteps=10000000,
            progress_bar=True,
            log_interval=1
        )
        
        # Save the final model
        model.save("moon_lander_ppo_final")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save("moon_lander_ppo_interrupted")
    
    return model

if __name__ == '__main__':
    train_ppo()
