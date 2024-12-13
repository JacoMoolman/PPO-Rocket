import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
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

class RewardLoggerCallback(BaseCallback):
    def __init__(self, num_envs, check_freq, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_rewards = {i: 0 for i in range(num_envs)}
        self.step_count = {i: 0 for i in range(num_envs)}
        self.total_steps = 0
        self.save_freq = 10000  # Save every 10k steps
        
    def _on_step(self) -> bool:
        self.total_steps += 1
        for env_idx in range(len(self.locals['rewards'])):
            self.total_rewards[env_idx] += self.locals['rewards'][env_idx]
            self.step_count[env_idx] += 1
            
            if self.step_count[env_idx] % self.check_freq == 0:
                with open(f'game{env_idx+1}.txt', 'a') as f:
                    f.write(f"{self.total_rewards[env_idx]}\n")
        
        if self.total_steps % self.save_freq == 0:
            # Save checkpoint
            self.model.save(f"moon_lander_checkpoint_{self.total_steps}")
            print(f"\nSaved checkpoint at {self.total_steps} steps")
        
        return True

def train_ppo(num_envs=10):  
    # Create multiple environments for parallel training
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Create the PPO model with custom parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
                vf=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
            )
        )
    )
    
    # Create and add our custom callback
    reward_logger = RewardLoggerCallback(num_envs=num_envs, check_freq=1000)
    
    try:
        # Train the agent
        model.learn(
            total_timesteps=10000000,
            callback=reward_logger,
            progress_bar=True,
            log_interval=1
        )
        
        # Save the final model
        model.save("moon_lander_ppo_final")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up environments...")
        env.close()  # Close all subprocesses
        print("Saving model...")
        model.save("moon_lander_ppo_interrupted")
    
    return model

if __name__ == '__main__':
    train_ppo()
