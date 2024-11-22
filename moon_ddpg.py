import torch
import numpy as np
from GetMoonGame_new import MoonLanderGame
from copy import deepcopy
from collections import deque
import torch.nn as nn
import random
import pygame
import sys

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.1, sigma=0.3):  
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state.flatten()  # Ensure 1D array

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim, output_tanh):
        super().__init__()
        # Wider and deeper network
        self.layer1 = nn.Linear(input_dim, layer1_dim)
        self.bn1 = nn.BatchNorm1d(num_features=layer1_dim, affine=True)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.bn2 = nn.BatchNorm1d(num_features=layer2_dim, affine=True)
        self.layer3 = nn.Linear(layer2_dim, layer2_dim)  # New layer
        self.bn3 = nn.BatchNorm1d(num_features=layer2_dim, affine=True)
        self.layer4 = nn.Linear(layer2_dim, layer2_dim // 2)  # New layer
        self.bn4 = nn.BatchNorm1d(num_features=layer2_dim // 2, affine=True)
        self.output_layer = nn.Linear(layer2_dim // 2, output_dim)
        
        self.output_tanh = output_tanh
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Handle single samples by adding batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
        elif len(input.shape) == 0:
            input = input.unsqueeze(0).unsqueeze(0)
            
        x = self.layer1(input)
        x = self.leaky_relu(x)
        if x.shape[0] > 1:  # Only apply BatchNorm for batch size > 1
            x = self.bn1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.leaky_relu(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.leaky_relu(x)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.leaky_relu(x)
        if x.shape[0] > 1:
            x = self.bn4(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        if self.output_tanh:
            x = self.tanh(x)
        return x

class DDPG():
    def __init__(self, state_dim, action_dim, action_scale, noise_decrease,
                 gamma=0.99, batch_size=256, q_lr=3e-4, pi_lr=1e-4, tau=0.001, memory_size=200000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Actor (Policy) network
        self.pi_model = NeuralNetwork(self.state_dim, 512, 384, self.action_dim, output_tanh=True).to(self.device)
        # Critic (Value) network
        self.q_model = NeuralNetwork(self.state_dim + self.action_dim, 512, 384, 1, output_tanh=False).to(self.device)
        
        # Target networks
        self.pi_target_model = deepcopy(self.pi_model).to(self.device)
        self.q_target_model = deepcopy(self.q_model).to(self.device)
        
        self.noise = OUNoise(self.action_dim)
        self.noise_threshold = 1
        self.min_noise = 0.4  
        self.noise_decrease = noise_decrease * 0.5  
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state):
        self.pi_model.eval()  # Set to evaluation mode
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            pred_action = self.pi_model(state_tensor).detach().cpu().numpy()
            
        self.pi_model.train()  # Set back to training mode
        # Add noise for exploration
        noise = self.noise.sample()
        noisy_action = pred_action + noise * self.noise_threshold
        # Clip actions to [-1, 1]
        clipped = np.clip(noisy_action, -1, 1)
        return clipped.flatten()  # Convert from (1,3) to (3,)

    def fit(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert batch to numpy arrays first
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        dones = np.array([x[3] for x in batch])
        next_states = np.array([x[4] for x in batch])
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)

        # Update critic
        self.q_model.train()
        self.pi_model.eval()
        with torch.no_grad():
            next_actions = self.pi_target_model(next_state_batch)
            target_q = self.q_target_model(torch.cat([next_state_batch, next_actions], dim=1))
            target_q = reward_batch + self.gamma * target_q * (1 - done_batch)
        
        current_q = self.q_model(torch.cat([state_batch, action_batch], dim=1))
        q_loss = torch.mean((current_q - target_q.detach()) ** 2)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update actor
        self.q_model.eval()
        self.pi_model.train()
        pred_actions = self.pi_model(state_batch)
        pi_loss = -torch.mean(self.q_model(torch.cat([state_batch, pred_actions], dim=1)))
        
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.q_target_model.parameters(), self.q_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.pi_target_model.parameters(), self.pi_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Keep minimum noise level
        self.noise_threshold = max(self.min_noise, self.noise_threshold - self.noise_decrease)

    def save_model(self, filepath):
        torch.save({
            'pi_model': self.pi_model.state_dict(),
            'q_model': self.q_model.state_dict(),
            'pi_target_model': self.pi_target_model.state_dict(),
            'q_target_model': self.q_target_model.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.pi_model.load_state_dict(checkpoint['pi_model'])
        self.q_model.load_state_dict(checkpoint['q_model'])
        self.pi_target_model.load_state_dict(checkpoint['pi_target_model'])
        self.q_target_model.load_state_dict(checkpoint['q_target_model'])

def draw_graph(surface, data, pos, size, color, title, max_val=None):
    if not data:
        return
    
    # Ensure data is a list of numbers
    data = [float(x) for x in data]
    
    # Draw points
    points = []
    for i, val in enumerate(data):
        x = pos[0] + (i / (len(data) - 1 if len(data) > 1 else 1)) * size[0]
        if max_val is None:
            max_val = max(max(data), abs(min(data))) if data else 1
        max_val = max(max_val, 0.1)  # Prevent division by zero
        min_val = min(data) if data else 0
        scaled_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        y = pos[1] + size[1] - (scaled_val * size[1])
        points.append((int(x), int(y)))
    
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)
        
    # Draw title
    font = pygame.font.Font(None, 24)
    text = font.render(title, True, color)
    surface.blit(text, (pos[0], pos[1] - 20))

def train_ddpg():
    # Initialize DDPG agent with adjusted parameters
    agent = DDPG(state_dim=15, action_dim=3, action_scale=1, 
                 noise_decrease=0.000001/3,  # Adjusted for 3 rockets
                 gamma=0.99,  
                 batch_size=256,  # Increased batch size
                 q_lr=3e-4,  # Slightly lower learning rates for stability
                 pi_lr=1e-4,
                 tau=0.001)  # Slower target network updates

    # Create custom neural network wrapper for the game
    class DDPGWrapper:
        def __init__(self, agent):
            self.agent = agent
            self.current_state = None
        
        def activate(self, inputs):
            state = np.array(inputs)
            actions = self.agent.get_action(state)
            self.current_state = state
            return [float(actions[0] > 0), float(actions[1] > 0), float(actions[2] > 0)]
        
        def learn(self, reward, next_state, done):
            if done and next_state is None:  # Window was closed
                return
            if isinstance(next_state, list):
                next_state = next_state[0]  # Take the first lander's state
            if isinstance(reward, list):
                reward = reward[0]  # Take the first lander's reward
            if isinstance(done, list):
                done = done[0]  # Take the first lander's done flag
            next_action = self.agent.get_action(next_state)
            self.agent.fit(self.current_state, next_action, reward, done, next_state)
    
    # Create game instance with DDPG wrapper
    net = DDPGWrapper(agent)
    game = MoonLanderGame(net)
    
    # Setup data for plotting
    game.training_data = {'scores': [], 'rewards': [], 'avg_rewards': []}
    game.draw_graph = draw_graph
    
    # Run the game continuously
    while True:
        # Run one step of the game
        state, reward, done = game.run_step()
        
        # Check if window was closed
        if done and state is None:
            print("Game window closed. Exiting...")
            sys.exit(0)
        
        # Let the agent learn from the experience
        net.learn(reward, state, done)
        
        # Update progress and graphs every 10 steps for smoother visualization
        if game.steps % 10 == 0:
            # Get the first lander's reward if it's a list
            current_reward = reward[0] if isinstance(reward, list) else reward
            
            # Update plot data
            game.training_data['scores'].append(float(game.total_score))
            game.training_data['rewards'].append(float(current_reward))
            
            # Calculate moving average of rewards
            window_size = 100
            if len(game.training_data['rewards']) >= window_size:
                avg_reward = sum(game.training_data['rewards'][-window_size:]) / window_size
                game.training_data['avg_rewards'].append(avg_reward)
            
            # Draw current reward graph (orange)
            draw_graph(game.screen, 
                      game.training_data['rewards'][-200:],
                      (600, 20), (180, 80),
                      (255, 165, 0),
                      "Current Reward",
                      max_val=500)
            
            # Draw average reward graph (green)
            if len(game.training_data['avg_rewards']) > 0:
                draw_graph(game.screen,
                          game.training_data['avg_rewards'][-200:],
                          (600, 130), (180, 80),
                          (0, 255, 0),
                          "Average Reward")
            
            # Update display
            pygame.display.flip()
        
        # Save model periodically
        if game.steps % 1000 == 0:
            agent.save_model('moon_lander_continuous.pth')
            print(f"Steps: {game.steps}, Total Score: {game.total_score}, Current Score: {game.score}, Last Reward: {current_reward:.2f}")

if __name__ == '__main__':
    train_ddpg()
