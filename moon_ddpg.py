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
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.5):  
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim, output_tanh):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.layer3 = nn.Linear(layer2_dim, output_dim)
        self.output_tanh = output_tanh
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.layer1(input)
        hidden = self.leaky_relu(hidden)
        hidden = self.layer2(hidden)
        hidden = self.leaky_relu(hidden)
        output = self.layer3(hidden)
        if self.output_tanh:
            return self.tanh(output)
        else:
            return output

class DDPG():
    def __init__(self, state_dim, action_dim, action_scale, noise_decrease,
                 gamma=0.99, batch_size=128, q_lr=1e-3, pi_lr=1e-4, tau=0.005, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Actor (Policy) network
        self.pi_model = NeuralNetwork(self.state_dim, 400, 300, self.action_dim, output_tanh=True).to(self.device)
        # Critic (Value) network
        self.q_model = NeuralNetwork(self.state_dim + self.action_dim, 400, 300, 1, output_tanh=False).to(self.device)
        
        # Target networks
        self.pi_target_model = deepcopy(self.pi_model).to(self.device)
        self.q_target_model = deepcopy(self.q_model).to(self.device)
        
        self.noise = OUNoise(self.action_dim)
        self.noise_threshold = 1
        self.min_noise = 0.3  # Minimum noise level
        self.noise_decrease = noise_decrease
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        pred_action = self.pi_model(state_tensor).detach().cpu().numpy()
        action = pred_action + self.noise_threshold * self.noise.sample()
        return action

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
        next_actions = self.pi_target_model(next_state_batch)
        target_q = self.q_target_model(torch.cat([next_state_batch, next_actions], dim=1))
        target_q = reward_batch + self.gamma * target_q * (1 - done_batch)
        current_q = self.q_model(torch.cat([state_batch, action_batch], dim=1))
        q_loss = torch.mean((current_q - target_q.detach()) ** 2)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update actor
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

# Configuration for graph display
GRAPH_HISTORY_LENGTH = 500  # Number of data points to show in the scrolling graph

def train_ddpg():
    def draw_graph(surface, data, pos, size, color, title, max_val=None):
        if not data:
            return
            
        # Only use the last GRAPH_HISTORY_LENGTH points
        display_data = data[-GRAPH_HISTORY_LENGTH:] if len(data) > GRAPH_HISTORY_LENGTH else data
            
        # Draw border and background
        pygame.draw.rect(surface, (50, 50, 50), (*pos, *size))
        pygame.draw.rect(surface, (200, 200, 200), (*pos, *size), 1)
        
        # Draw title
        font = pygame.font.Font(None, 20)  # Smaller font
        text = font.render(title, True, (200, 200, 200))
        surface.blit(text, (pos[0], pos[1] - 15))  # Position just above graph
        
        # Calculate scaling
        if max_val is None:
            max_val = max(max(display_data), abs(min(display_data))) if display_data else 1
        max_val = max(max_val, 0.1)  # Prevent division by zero
        min_val = min(display_data) if display_data else 0
        
        # Draw y-axis labels
        font_small = pygame.font.Font(None, 16)
        # Max value
        max_label = font_small.render(f"{int(max_val)}", True, (200, 200, 200))
        surface.blit(max_label, (pos[0] - 25, pos[1]))
        # Min value
        min_label = font_small.render(f"{int(min_val)}", True, (200, 200, 200))
        surface.blit(min_label, (pos[0] - 25, pos[1] + size[1] - 10))
        
        # Draw points - use all points and scale x-axis to fit
        points = []
        for i, val in enumerate(display_data):
            x = pos[0] + (i / (len(display_data) - 1 if len(display_data) > 1 else 1)) * size[0]
            scaled_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            y = pos[1] + size[1] - (scaled_val * size[1])
            points.append((int(x), int(y)))
            
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)

    # Initialize DDPG agent with adjusted parameters
    agent = DDPG(state_dim=15, action_dim=3, action_scale=1, 
                 noise_decrease=0.00001,  
                 gamma=0.99,  
                 batch_size=128,  
                 q_lr=1e-3,  
                 pi_lr=1e-4,
                 tau=0.005)  
    
    # Create custom neural network wrapper for the game
    class DDPGWrapper:
        def __init__(self, agent):
            self.agent = agent
        
        def activate(self, inputs):
            state = np.array(inputs)
            actions = self.agent.get_action(state)
            self.current_state = state
            return [float(actions[0] > 0), float(actions[1] > 0), float(actions[2] > 0)]
        
        def learn(self, reward, next_state, done):
            if done and next_state is None:  # Window was closed
                return
            next_action = self.agent.get_action(next_state)
            self.agent.fit(self.current_state, next_action, reward, done, next_state)
    
    # Create game instance with DDPG wrapper
    net = DDPGWrapper(agent)
    game = MoonLanderGame(net)
    
    # Setup data for plotting
    game.training_data = {'rewards': [], 'last_rewards': []}
    game.draw_graph = draw_graph
    
    # Run the game continuously
    while True:
        # Run one step of the game
        state, reward, done = game.run_step()
        game.update_noise(agent.noise_threshold)  # Update noise value in game
        
        # Check if window was closed
        if done and state is None:
            print("Game window closed. Exiting...")
            sys.exit(0)
        
        # Let the agent learn from the experience
        net.learn(reward, state, done)
        
        # Update progress every 10 steps
        if game.steps % 10 == 0:
            agent.save_model('moon_lander_continuous.pth')
            print(f"Steps: {game.steps}, Total Rewards: {game.total_rewards:.2f}, Last Reward: {reward:.2f}")
            
            # Update plot data - keep all points
            game.training_data['rewards'].append(game.total_rewards)
            game.training_data['last_rewards'].append(reward)
            
            game.update_noise(agent.noise_threshold)  # Update noise value in game
            pygame.display.update()

if __name__ == '__main__':
    train_ddpg()
