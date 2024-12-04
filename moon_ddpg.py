import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import pygame
from GetMoonGame_new import MoonLanderGame

class MoonLanderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        # Create a dummy network for MoonLanderGame (it won't be used since we'll override the actions)
        class DummyNetwork:
            def activate(self, x):
                return [0, 0, 0]
        
        self.game = MoonLanderGame(DummyNetwork())
        
    def step(self, action):
        # Convert continuous actions to discrete for rotation and thrust
        discrete_action = [
            float(action[0] > 0.3),  # Rotate right if > 0.3
            float(action[1] > 0.3),  # Rotate left if > 0.3
            float(action[2] > 0)     # Thrust if > 0
        ]
        
        # Override the network's activate function
        self.game.net.activate = lambda x: discrete_action
        
        # Run one step of the game
        state, reward, done = self.game.run_step()
        
        if state is None:  # Game quit
            return np.zeros(15), 0, True, True, {}
            
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
            0,  # normalized_angle (will be updated in first step)
            0,  # normalized_angular_velocity
            0,  # relative_velocity_x
            0,  # relative_velocity_y
            min(self.game.position.x, self.game.WIDTH - self.game.position.x) / self.game.WIDTH,
            min(self.game.position.y, self.game.HEIGHT - self.game.position.y) / self.game.HEIGHT,
            0,  # angle_between_direction_velocity
        ]
        return np.array(state, dtype=np.float32), {}
    
    def render(self):
        # Rendering is handled by the game class
        pass
    
    def close(self):
        pygame.quit()

def train_ddpg():
    # Create the environment
    env = MoonLanderEnv()
    
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Create the DDPG model
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        tau=0.005
    )
    
    try:
        # Train the agent
        model.learn(total_timesteps=1000000)
        
        # Save the trained model
        model.save("moon_lander_ddpg")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save("moon_lander_ddpg_interrupted")
    
    finally:
        env.close()

if __name__ == '__main__':
    train_ddpg()
