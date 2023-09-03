import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Polygon

from Classes2.Shapes import Room, Obstacle
from Classes2.Agent import Agent
from Classes2.Sensors import UWBAnchor
from Classes2.VoronoiHandler import VoronoiHandler
from Classes2.RobotAssigner import RobotAssigner

class Environment:
    def __init__(self, filename):
        self.filename = filename
        self.agents = []
        self.room = None
        self.anchors = []
        self.shapes_coord = []
        self.importCSV()
        # create a VoronoiHandler object
        self.vh = None
        # create a RobotAssigner object
        self.ra = None
        # create variables to save regions and paths
        self.region_paths = []
        # create handles for the figure and axes
        self.fig, self.axes = None, None
        
    def importCSV(self):
        df = pd.read_csv(self.filename, header=None, skiprows=1, na_values=['NA', 'na'])
      
        for i in range(len(df)):
            x = np.fromstring(df.iloc[i, 1], sep=' ')
            y = np.fromstring(df.iloc[i, 2], sep=' ') * -1
            element_type = df.iloc[i, 4]
            self.shapes_coord.append([x, y, element_type])

            if element_type == 'room':
                x_max = np.max(x)
                y_max = np.max(y)
                x_min = np.min(x)
                y_min = np.min(y)
        width_room = abs(x_max - x_min)
        height_room = abs(y_max - y_min)

        x_scale = [0, width_room]
        y_scale = [0, height_room]

        # normalize the vertices of the room, obstacles and agents
        for j in range(len(self.shapes_coord[0])-1):
            for i in range(len(self.shapes_coord)):
                if j == len(self.shapes_coord[0]) - 2:
                    self.shapes_coord[i][j] = (self.shapes_coord[i][j] - y_min) * (y_scale[1] - y_scale[0]) / (y_max - y_min) + y_scale[0]
                else:
                    self.shapes_coord[i][j] = (self.shapes_coord[i][j] - x_min) * (x_scale[1] - x_scale[0]) / (x_max - x_min) + x_scale[0]
        for i in range(len(self.shapes_coord)):

            if self.shapes_coord[i][2] == 'room':
                self.room = Room(self.shapes_coord[i][0], self.shapes_coord[i][1])
    
                # add UWB anchors to the vertices of the room
                for j in range(len(self.room.x)):
                    self.addUWBAnchor(self.room.x[j], self.room.y[j])

            elif self.shapes_coord[i][2] == 'obstacle':
                obstacle = Obstacle(self.shapes_coord[i][0], self.shapes_coord[i][1])
                self.room.addObstacle(obstacle)

            elif self.shapes_coord[i][2] == 'agent':
                agent = Agent(self.shapes_coord[i][0], self.shapes_coord[i][1])
                self.agents.append(agent)
            else:
                raise ValueError(f"Unknown element type: {element_type}")
        
    def addUWBAnchor(self, x, y):
        anchor = UWBAnchor(x, y)
        self.anchors.append(anchor)
        return anchor
    
    def createVoronoiTessellation(self, num_points=300):
        self.vh = VoronoiHandler([self.room.x, self.room.y])
        self.vh.generate_voronoi_within_room(num_points)
    
    def assignRobots(self):
        self.ra = RobotAssigner(self.vh.vor, [self.room.x, self.room.y], len(self.agents))
        self.ra.divide_areas_using_kmeans()
        self.region_paths = self.ra.compute_tsp_paths()

    def multilaterationUWB(self, agent, distances):
        result = minimize(self.cost_function, [agent.x_real, agent.y_real], args=(self.anchors, distances, np.ones(len(distances))), method='Nelder-Mead')
        return np.array([result.x[0], result.x[1]])
    
    def cost_function(self, initial_guess, anchors, distances, weights):
        eq = []
        for i in range(len(distances)):
            eq.append((np.sqrt((anchors[i].x - initial_guess[0])**2 + (anchors[i].y - initial_guess[1])**2) - distances[i])*weights[i])
        return np.sum(np.array(eq)**2)
    
    def createAgents(self, num_agents):
        if num_agents > 1:
            for i in range(num_agents-1):
                self.agents.append(Agent(self.agents[0].x, self.agents[0].y))

    def plotRoom(self, ax):
        ax.set_title('Room')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        ax.plot(self.room.x, self.room.y, 'k', linewidth=1)
        for obstacle in self.room.obstacles:
            ax.plot(obstacle.x, obstacle.y, 'k', linewidth=1)
        for anchor in self.anchors:
            ax.plot(anchor.x, anchor.y, 'o', markersize=4, markerfacecolor='r', markeredgecolor='r')
    
    def plotVoronoiTessellation(self, ax):
        ax.set_title('Voronoi Tessellation')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        for line in self.vh.clipped_ridges:
            x, y = line.xy
            ax.plot(x, y, color="black", linewidth=0.5)
        
    def plotAgentAssignments(self, ax):
        ax.set_title('Agents Regions')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        for i in range(len(self.ra.robot_assignments)):
            for region in self.ra.robot_assignments[i]:
                # Check if the region is a MultiPolygon
                if region.geom_type == 'MultiPolygon':
                    for sub_region in region.geoms:
                        x, y = sub_region.exterior.xy
                        ax.fill(x, y, color='C'+str(i), alpha=0.5)
                else:
                    x, y = region.exterior.xy
                    ax.fill(x, y, color='C'+str(i), alpha=0.5)
        
    def plotAgentAssignmentsAndPaths(self, ax):
        ax.set_title('Agents Regions with TSP Paths')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        for i in range(len(self.region_paths)):
            ax.plot(self.region_paths[i][0], self.region_paths[i][1], marker='o', markersize=2, linewidth=1, color='C'+str(i), label=f"Robot {i}")
        # plot room boundary
        ax.plot(self.room.x, self.room.y, color="k", linewidth=1, label="Room")

    def simulate(self, ax, dt=0.1):
        k=0
        while True:
            # Clear the axes and redraw the environment of the first plot
            ax.cla()
            self.plotRoom(ax)
            # Draw the agents as polygons
            for i in range(len(self.agents)):
                if k >= len(self.region_paths[i][0]):
                    break
                # place the agent at the beginning of the tsp path
                initalCoM = np.array([self.region_paths[i][0][k], self.region_paths[i][1][k]])
                
                # move the coordinates of the agent 
                self.agents[i].x = initalCoM[0] + self.agents[i].x_ideal - self.agents[i].CoM_ideal[0]
                self.agents[i].y = initalCoM[1] + self.agents[i].y_ideal - self.agents[i].CoM_ideal[1]
                # plot the agent
                ax.add_patch(Polygon(np.array([self.agents[i].x, self.agents[i].y]).T, color='C'+str(i), alpha=0.5))
            k=k+1
            # if window is closed, then stop the simulation
            if not plt.fignum_exists(ax.get_figure().number):
                break
            plt.pause(dt)
    