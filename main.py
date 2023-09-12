from Classes.Environment import Environment
import matplotlib.pyplot as plt
import datetime

num_agents = 10
filename = 'noObstacles'
env = Environment('Rooms/'+filename+'.csv')
env.createAgents(num_agents)
env.createVoronoiTessellation(100)
env.assignRobots()
env.initializeAgentSensors(['Encoders', 'Magnetometer', 'Lidar', 'StereoCamera'])

# create a figure with 4 subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
# set size 1280x720 pixels with dpi=100
fig.set_size_inches(12.8, 7.2)

env.plotRoom(axes[0, 0])
env.plotVoronoiTessellation(axes[0, 1])
env.plotAgentAssignments(axes[0, 1])
env.plotAgentAssignmentsAndPaths(axes[1, 0])

# video file name
video_name = 'videos/'+filename+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# start the simulation
env.simulate(axes[1, 1], dt=0.1, saveVideo=False, videoName=video_name, videoSpeed=1.0)
