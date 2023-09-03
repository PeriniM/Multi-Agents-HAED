import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Polygon

from Classes.Shapes import Room, Obstacle, Agent
from Classes.Sensors import UWBAnchor

class Environment:
    def __init__(self, filename):
        self.filename = filename
        self.agents = []
        self.room = None
        self.obstacles = []
        self.anchors = []
        self.shapes_coord = []
        self.map_real_vert = []
        self.agents_real_CoM = []
        self.fig, self.axes = plt.subplots(nrows=2, ncols=1)
        self.importCSV()
        
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
        aspect_ratio = width_room / height_room

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
                self.map_real_vert.append([self.room.x, self.room.y])
                # add UWB anchors to the vertices of the room
                for j in range(len(self.room.x)):
                    self.addUWBAnchor(self.room.x[j], self.room.y[j])

            elif self.shapes_coord[i][2] == 'obstacle':
                obstacle = Obstacle(self.shapes_coord[i][0], self.shapes_coord[i][1])
                self.obstacles.append(obstacle)
                self.map_real_vert.append([obstacle.x, obstacle.y])

            elif self.shapes_coord[i][2] == 'agent':
                agent = Agent(self.shapes_coord[i][0], self.shapes_coord[i][1])
                self.agents.append(agent)
                self.agents_real_CoM.append(agent.getRealCoM())

            else:
                raise ValueError(f"Unknown element type: {element_type}")
        
    def addUWBAnchor(self, x, y):
        anchor = UWBAnchor(x, y)
        self.anchors.append(anchor)
        return anchor
    
    def multilaterationUWB(self, agent, distances):
        result = minimize(self.cost_function, [agent.x_real, agent.y_real], args=(self.anchors, distances, np.ones(len(distances))), method='Nelder-Mead')
        return np.array([result.x[0], result.x[1]])
    
    def cost_function(self, initial_guess, anchors, distances, weights):
        eq = []
        for i in range(len(distances)):
            eq.append((np.sqrt((anchors[i].x - initial_guess[0])**2 + (anchors[i].y - initial_guess[1])**2) - distances[i])*weights[i])
        return np.sum(np.array(eq)**2)
    
    def duplicateAgents(self, num_agents):
        for i in range(num_agents):
            self.agents.append(Agent(self.agents[0].x, self.agents[0].y))
            self.agents_real_CoM.append(self.agents[0].getRealCoM())

    def simulate(self, dt):
        while True:
            # Clear the axes and redraw the environment of the first plot
            self.axes[0].cla()
            self.axes[0].set_title('Ideal Environment')
            self.axes[0].set_xlabel('x (m)')
            self.axes[0].set_ylabel('y (m)')
            self.axes[0].set_aspect('equal', adjustable='box')
            self.axes[0].grid(True, which='both', linestyle='dotted')
            # Draw the room
            self.axes[0].plot(self.room.x, self.room.y, 'k', linewidth=1)
            # Draw the obstacles
            for obstacle in self.obstacles:
                self.axes[0].plot(obstacle.x, obstacle.y, 'k', linewidth=1)
            # Draw the agents as polygons
            for agent in self.agents:
                self.axes[0].add_patch(Polygon(np.array([agent.x, agent.y]).T, closed=True, fill=True, facecolor='b', edgecolor='b', alpha=0.7))
            # Draw the anchors
            for anchor in self.anchors:
                self.axes[0].plot(anchor.x, anchor.y, 'o', markersize=4, markerfacecolor='r', markeredgecolor='r')
        
            # Clear the axes and redraw the environment of the second plot
            self.axes[1].cla()
            self.axes[1].set_title('Real Environment')
            self.axes[1].set_xlabel('x (m)')
            self.axes[1].set_ylabel('y (m)')
            self.axes[1].set_aspect('equal', adjustable='box')
            self.axes[1].grid(True, which='both', linestyle='dotted')

            # Draw the anchors
            for anchor in self.anchors:
                self.axes[1].plot(anchor.x, anchor.y, 'o', markersize=4, markerfacecolor='r', markeredgecolor='r')

            # for each agent
            for agent in self.agents:
                # ------------------ CALCULATIONS ------------------

                # UWB MULTILATERATION
                # get the distances from the anchors
                distances = []
                for anchor in self.anchors:
                    if agent.CoM_estim is None:
                        distances.append(anchor.getDistance([agent.x_real, agent.y_real]))
                    else:
                        distances.append(anchor.getDistance(agent.CoM_estim))
                # get the coordinates of the agent
                agent.CoM_estim = self.multilaterationUWB(agent, distances)
                # ------------------ DYNAMICS ------------------
                # update the dynamics of the agent
                agent.updateDynamics(dt)

                # ------------------ PLOTTING ------------------
                # draw the estimates CoM of the agent
                self.axes[1].plot(agent.CoM_estim[0], agent.CoM_estim[1], 'o', markersize=4, markerfacecolor='b', markeredgecolor='b')
            plt.pause(dt)

    def getFigureAndAxes(self):
        return self.fig, self.axes
    