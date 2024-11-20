import torch
import numpy as np
from GetMoonGame_new import MoonLanderGame
from copy import deepcopy
from collections import deque
import torch.nn as nn
import random

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
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
                 gamma=0.999, batch_size=64, q_lr=1e-4, pi_lr=1e-5, tau=0.001, memory_size=100000):
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

        self.noise_threshold = max(0.1, self.noise_threshold - self.noise_decrease)

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

def train_ddpg():
    # Initialize DDPG agent
    # State dimension: 15 (from the game's neural network inputs)
    # Action dimension: 3 (rotate left, rotate right, thrust)
    agent = DDPG(state_dim=15, action_dim=3, action_scale=1, noise_decrease=0.0001)
    
    # Create a custom neural network wrapper for the game
    class DDPGWrapper:
        def __init__(self, agent):
            self.agent = agent
        
        def activate(self, inputs):
            # Convert inputs to numpy array
            state = np.array(inputs)
            # Get action from DDPG agent
            actions = self.agent.get_action(state)
            # Store the current state for learning
            self.current_state = state
            # Convert continuous actions to binary decisions
            return [float(actions[0] > 0), float(actions[1] > 0), float(actions[2] > 0)]
        
        def learn(self, reward, next_state, done):
            # Get the next action using the target policy
            next_action = self.agent.get_action(next_state)
            # Store experience and learn
            self.agent.fit(self.current_state, next_action, reward, done, next_state)
    
    # Create game instance with DDPG wrapper
    net = DDPGWrapper(agent)
    game = MoonLanderGame(net)
    
    # Run the game continuously
    while True:
        # Run one step of the game
        state, reward, done = game.run_step()
        
        # Let the agent learn from the experience
        net.learn(reward, state, done)
        
        # Save the model periodically and show progress
        if game.steps % 1000 == 0:
            agent.save_model('moon_lander_continuous.pth')
            print(f"Steps: {game.steps}, Total Score: {game.total_score}, Current Score: {game.score}, Last Reward: {reward:.2f}")

if __name__ == '__main__':
    train_ddpg()
