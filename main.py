from Classes2.Environment import Environment
import matplotlib.pyplot as plt

num_agents = 6
env = Environment('Rooms/dungeon.csv')
env.createAgents(num_agents)
env.createVoronoiTessellation(200)
env.assignRobots()
env.initializeAgentSensors(['Encoders', 'Lidar', 'StereoCamera'])

# create a figure with 4 subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

env.plotRoom(axes[0, 0])
env.plotVoronoiTessellation(axes[0, 1])
env.plotAgentAssignments(axes[0, 1])
env.plotAgentAssignmentsAndPaths(axes[1, 0])

# start the simulation
env.simulate(axes[1, 1], dt=0.1)
