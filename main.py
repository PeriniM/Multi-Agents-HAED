import matplotlib.pyplot as plt
from Classes.Environment import Environment

# Example usage
env = Environment()
env.importCSV('Rooms/createGrid().csv')

# Create and add agents to the environment
env.duplicateAgents(2)
# Update the plot and simulate the environment
env.simulate(0.01)
