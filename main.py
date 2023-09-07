from Classes.Environment import Environment
import matplotlib.pyplot as plt

num_agents = 8
env = Environment('Rooms/dungeon.csv')
env.createAgents(num_agents)
env.createVoronoiTessellation(100)
env.assignRobots()
env.initializeAgentSensors(['Encoders', 'Lidar', 'StereoCamera'])

# create a figure with 4 subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
# set size 1920x1080
fig.set_size_inches(19.2, 10.8)
# full screen figure
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()

env.plotRoom(axes[0, 0])
env.plotVoronoiTessellation(axes[0, 1])
env.plotAgentAssignments(axes[0, 1])
env.plotAgentAssignmentsAndPaths(axes[1, 0])

# start the simulation
env.simulate(axes[1, 1], dt=0.1, saveVideo=False, videoName='dungeon', videoSpeed=1.0)
