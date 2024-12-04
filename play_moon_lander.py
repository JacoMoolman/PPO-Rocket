import gymnasium as gym
from stable_baselines3 import PPO
from moon_ppo import MoonLanderEnv
import pygame
import time

def play_moon_lander(model_path="best_moon_lander.pth"):
    # Create the environment
    env = MoonLanderEnv()
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Play episodes
    while True:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get action from the trained model
            action, _ = model.predict(obs)
            
            # Take step in environment
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Control game speed
            time.sleep(1/60)  # 60 FPS
            
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
            # Handle keyboard input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:  # Press 'Q' to quit
                pygame.quit()
                return
            
        print(f"Episode finished with reward: {total_reward}")

if __name__ == "__main__":
    play_moon_lander()
