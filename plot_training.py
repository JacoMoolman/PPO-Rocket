import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation
import os

class LivePlotter:
    def __init__(self):
        # Setup the figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('DDPG Training Progress')
        
        # Initialize empty data
        self.steps_data = []
        self.scores_data = []
        self.rewards_data = []
        
        # Setup the lines
        self.score_line, = self.ax1.plot([], [], 'g-', label='Total Score')
        self.reward_line, = self.ax2.plot([], [], 'b-', label='Last Reward')
        
        # Setup the axes
        self.ax1.set_ylabel('Total Score')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Last Reward')
        self.ax2.grid(True)
        self.ax2.legend()
        
        plt.tight_layout()
        
    def update(self, frame):
        try:
            # Read the latest data
            if os.path.exists('training_data.json'):
                with open('training_data.json', 'r') as f:
                    data = json.load(f)
                    
                self.steps_data = data['steps']
                self.scores_data = data['scores']
                self.rewards_data = data['rewards']
                
                # Update the lines
                self.score_line.set_data(self.steps_data, self.scores_data)
                self.reward_line.set_data(self.steps_data, self.rewards_data)
                
                # Adjust the view
                if len(self.steps_data) > 0:
                    self.ax1.relim()
                    self.ax1.autoscale_view()
                    self.ax2.relim()
                    self.ax2.autoscale_view()
        except:
            pass
        
        return self.score_line, self.reward_line

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, interval=100, blit=True)
        plt.show()

if __name__ == "__main__":
    plotter = LivePlotter()
    plotter.animate()
