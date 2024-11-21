import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import time

class LivePlotter:
    def __init__(self):
        # Setup the figure and subplots
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('DDPG Training Progress')
        
        # Initialize empty lines
        self.score_line, = self.ax1.plot([], [], 'g-', label='Total Score')
        self.reward_line, = self.ax2.plot([], [], 'b-', label='Last Reward')
        
        # Setup axes
        self.ax1.set_ylabel('Total Score')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Last Reward')
        self.ax2.grid(True)
        self.ax2.legend()
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Initialize data
        self.last_update = 0
        
    def update(self, frame):
        try:
            # Read CSV file
            df = pd.read_csv('training_data.csv')
            
            if len(df) == 0:
                return
            
            # Update data
            self.score_line.set_data(df['Steps'], df['Score'])
            self.reward_line.set_data(df['Steps'], df['Reward'])
            
            # Adjust limits
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # Draw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            pass

def main():
    plotter = LivePlotter()
    ani = FuncAnimation(plotter.fig, plotter.update, interval=1000)  # Update every second
    plt.show()

if __name__ == "__main__":
    main()
