import matplotlib.pyplot as plt
from Classes.Environment import Environment

# Example usage
env = Environment()
env.importCSV('Rooms/createGrid().csv')

# Create and add agents, shapes, and anchors to the environment

# Update the plot and simulate the environment
while True:
    env.updatePlot(type='ideal')
    env.updatePlot(type='real')
    plt.pause(0.01)
