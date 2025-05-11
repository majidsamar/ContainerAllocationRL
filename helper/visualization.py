import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class YardVisualizer:
    def __init__(self, bays, rows, tiers):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
    def set_yard(self, yard):
        self.yard = yard

    def draw_yard(self):
        import matplotlib.pyplot as plt
        from matplotlib import ticker

        fig = plt.figure(figsize=(8, 4))
        self.ax = fig.add_subplot(111, projection='3d')

        # Draw cubes
        for z in range(self.tiers):
            for y in range(self.rows):
                for x in range(self.bays):
                    if self.yard[z, y, x] != 0:
                        self.draw_cube_2x(x=x, y=y, z=z)
                        self.add_text_label(x=x, y=y, z=z, dwell_time=self.yard[z, y, x])

        # Axis labels with bold blue text
        self.ax.set_xlabel('BAY', labelpad=10, fontsize=12, color='blue', weight='bold')
        self.ax.set_ylabel('ROW', labelpad=10, fontsize=12, color='blue', weight='bold')
        self.ax.set_zlabel('TIER', labelpad=10, fontsize=12, color='blue', weight='bold')

        # Axis limits
        self.ax.set_xlim(0, 2 * self.bays + 1)
        self.ax.set_ylim(0, self.rows + 1)
        self.ax.set_zlim(0, self.tiers + 1)

        # Set grid ticks at even numbers only
        def even_ticks(max_val):
            return [i for i in range(0, max_val + 1) if i % 2 == 0]

        self.ax.set_xticks(even_ticks(2 * self.bays + 1))
        self.ax.set_yticks(even_ticks(self.rows + 1))
        self.ax.set_zticks(even_ticks(self.tiers + 1))

        # Optional: turn on grid lines manually for visibility
        self.ax.grid(True)

        # Set box aspect ratio
        self.ax.set_box_aspect([4 * self.bays, 4 * self.rows, 2 * self.tiers])

        plt.tight_layout()
        plt.show()

    def draw_cube(self, x, y, z, color=(0, 0, 1, 1)):
        # Define the vertices of the cube
        vertices = [
            [x, y, z],
            [x + 1, y, z],
            [x + 1, y + 1, z],
            [x, y + 1, z],
            [x, y, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1]
        ]

        # Define the six faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

        # Create a 3D polygon collection
        poly3d = Poly3DCollection(faces, alpha=.25, linewidths=0.1, edgecolors='r')
        poly3d.set_facecolor(color)
        self.ax.add_collection3d(poly3d)

    def draw_cube_2x(self, x, y, z, color=(0, 0, 1, 1)):
        # Define the vertices of the cube with scaling
        vertices = [
            [2 * x, y, z],
            [2 * (x + 1), y, z],
            [2 * (x + 1), y + 1, z],
            [2 * x, y + 1, z],
            [2 * x, y, z + 1],
            [2 * (x + 1), y, z + 1],
            [2 * (x + 1), y + 1, z + 1],
            [2 * x, y + 1, z + 1]
        ]

        # Define the six faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

        # Create a 3D polygon collection
        poly3d = Poly3DCollection(faces, alpha=.20, linewidths=0.1, edgecolors='r')
        poly3d.set_facecolor(color)
        self.ax.add_collection3d(poly3d)

    def add_text_label(self, x, y, z, dwell_time):
        """ Adds text label showing dwell time at the center of the cube. """
        self.ax.text(2 * x + 1, y + 0.5, z + 0.5, str(dwell_time), color='black',
                     fontsize=8, ha='center', va='center')

def plot_q_values(qMaxValues, qMinValues):
    plt.figure(figsize=(10, 5))
    plt.plot(qMaxValues, label="Max Q-value", color="blue")
    plt.plot(qMinValues, label="Min Q-value", color="red")
    plt.xlabel("Training Steps")
    plt.ylabel("Q-value")
    plt.title("Max and Min Q-values Over Training")
    plt.legend()
    plt.show()
def plot_learning_progress(rewards, losses, epsilons):
    """ Plots reward, loss, and epsilon decay over episodes. """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Episode Rewards
    axs[0].plot(rewards, label="Total Reward per Episode", color='blue')
    axs[0].set_title("Rewards Over Time")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Loss Values
    axs[1].plot(losses, label="Loss per Episode", color='red')
    axs[1].set_title("Loss Over Time")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Loss Value")
    axs[1].legend()
    axs[1].grid(True)

    # Plot Epsilon Decay
    axs[2].plot(epsilons, label="Epsilon Decay", color='green')
    axs[2].set_title("Epsilon Decay Over Time")
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("Epsilon")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
