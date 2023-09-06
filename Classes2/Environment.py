import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Polygon

from Classes2.Shapes import Room, Obstacle
from Classes2.Agent import Agent
from Classes2.Sensors import UWBAnchor, Encoder, Gyroscope, Accelerometer, Magnetometer, LiDAR
from Classes2.VoronoiHandler import VoronoiHandler
from Classes2.RobotAssigner import RobotAssigner

class Environment:
    def __init__(self, filename):
        self.filename = filename
        self.agents = []
        self.room = None
        self.anchors = []
        self.anchors_distances = []
        self.shapes_coord = []
        # create variable to store ideal map of the environment
        self.ideal_map = []
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
                self.ideal_map.append(self.room.getLines())
                # add UWB anchors to the vertices of the room
                for j in range(len(self.room.vertex_x)):
                    self.addUWBAnchor(self.room.vertex_x[j], self.room.vertex_y[j])

            elif self.shapes_coord[i][2] == 'obstacle':
                obstacle = Obstacle(self.shapes_coord[i][0], self.shapes_coord[i][1])
                self.room.addObstacle(obstacle)
                self.ideal_map.append(obstacle.getLines())

            elif self.shapes_coord[i][2] == 'agent':
                agent = Agent(self.shapes_coord[i][0], self.shapes_coord[i][1])
                self.agents.append(agent)
            else:
                raise ValueError(f"Unknown element type: {element_type}")
        
    def addUWBAnchor(self, x, y):
        anchor = UWBAnchor(x, y, noise_std_dev=0)
        self.anchors.append(anchor)
        return anchor
    
    def createVoronoiTessellation(self, num_points=300):
        self.vh = VoronoiHandler([self.room.vertex_x, self.room.vertex_y])
        self.vh.generate_voronoi_within_room(num_points)
    
    def assignRobots(self):
        self.ra = RobotAssigner(self.vh.vor, [self.room.vertex_x, self.room.vertex_y], len(self.agents))
        self.ra.divide_areas_using_kmeans()
        self.region_paths = self.ra.compute_tsp_paths()
        # assign the paths to the agents
        for i in range(len(self.agents)):
            self.agents[i].target_points = [self.region_paths[i][0], self.region_paths[i][1]]

    def createAgents(self, num_agents):
        if num_agents > 1:
            for i in range(num_agents-1):
                self.agents.append(Agent(self.agents[0].vertex_x, self.agents[0].vertex_y))
    
    def initializeAgentSensors(self):
        for agent in self.agents:
            left_encoder = Encoder(agent.x - agent.wheels_distance/2, agent.y, "EncoderLeft", 720, noise_std_dev=10)
            right_encoder = Encoder(agent.x + agent.wheels_distance/2, agent.y, "EncoderRight", 720, noise_std_dev=10)
            gyroscope = Gyroscope(agent.x, agent.y, noise_std_dev=0)
            accelerometer = Accelerometer(agent.x, agent.y, noise_std_dev=0)
            magnetometer = Magnetometer(agent.x, agent.y, noise_std_dev=0)
            lidar = LiDAR(agent.x, agent.y, "Lidar", 5, 90, 360, noise_std_dev=0)
            #stereo_camera = DepthSensor(agent.x, agent.y, "StereoCamera", 5, 240, 120, noise_std_dev=0)
            agent.initialize_sensors([left_encoder, right_encoder, gyroscope, accelerometer, magnetometer, lidar])

    def multilaterationUWB(self, agent, distances):
        result = minimize(self.cost_function, [agent.x, agent.y], args=(self.anchors, distances, np.ones(len(distances))), method='Nelder-Mead')
        return np.array([result.x[0], result.x[1]])
    
    def cost_function(self, initial_guess, anchors, distances, weights):
        eq = []
        for i in range(len(distances)):
            eq.append((np.sqrt((anchors[i].x - initial_guess[0])**2 + (anchors[i].y - initial_guess[1])**2) - distances[i])*weights[i])
        return np.sum(np.array(eq)**2)

    def simulate(self, ax, dt=0.1):

        # Initialize each agent's position to the starting point of its path
        for agent in self.agents:
            agent.x, agent.y = agent.target_points[0][0], agent.target_points[1][0]
            agent.theta = 0  # optionally reset orientation if needed

        while True:
            # Clear the axes and redraw the environment of the first plot
            ax.cla()
            self.plotRoom(ax)
            for idx, agent in enumerate(self.agents):
                # Only update the agent's position if there are remaining points in its path
                if len(agent.target_points[0]) > 1:
                    target = [agent.target_points[0][1], agent.target_points[1][1]]

                    #-----------------SENSORS-----------------
                    # UWB MULTILATERATION
                    self.anchors_distances = []
                    for anchor in self.anchors:
                        anchor.update(agent.x, agent.y)
                        self.anchors_distances.append(anchor.get_data())
                    # Update agent's position using multilateration
                    agent.estim_pos_uwb = self.multilaterationUWB(agent, self.anchors_distances)
                    agent.estim_v_uwb = np.array([agent.estim_pos_uwb[0] - agent.x, agent.estim_pos_uwb[1] - agent.y]) / dt
                    agent.estim_theta_uwb = np.arctan2(agent.estim_pos_uwb[1] - agent.y, agent.estim_pos_uwb[0] - agent.x)
                    agent.estim_omega_uwb = (agent.estim_theta_uwb - agent.theta) / dt
            
                    # ENCODERS
                    # Update encoder data for left and right wheels
                    agent.sensors["Encoder_left"].update(agent.wheel_radius, agent.v - agent.omega * agent.wheels_distance / 2, dt)
                    agent.sensors["Encoder_right"].update(agent.wheel_radius, agent.v + agent.omega * agent.wheels_distance / 2, dt)
                    # Estimate angular displacement and update encoders-based estimations
                    delta_theta_left = agent.get_sensor_data("Encoder_left") / agent.sensors["Encoder_left"].ticks_per_revolution * 2 * np.pi
                    delta_theta_right = agent.get_sensor_data("Encoder_right") / agent.sensors["Encoder_right"].ticks_per_revolution * 2 * np.pi
                    delta_theta = 0.5 * (delta_theta_right + delta_theta_left)
                    delta_s = agent.wheel_radius * delta_theta
                    delta_x = delta_s * np.cos(agent.theta + 0.5 * delta_theta)
                    delta_y = delta_s * np.sin(agent.theta + 0.5 * delta_theta)
                    # Calculate estimated position using encoder data
                    agent.estim_pos_encoders = np.array([agent.x + delta_x, agent.y + delta_y])
                    agent.estim_v_encoders = np.array([delta_x, delta_y]) / dt
                    agent.estim_omega_encoders = delta_theta / dt
                    agent.estim_theta_encoders = agent.theta + delta_theta

                    # LIDAR
                    # Update lidar data
                    agent.sensors["LiDAR"].update(agent.x, agent.y, agent.theta, self.ideal_map)
                    # Convert LiDAR distances to cartesian coordinates
                    lidar_distances = agent.sensors["LiDAR"].get_data()
                    lidar_cartesian_coords = agent.sensors["LiDAR"].convert_to_cartesian(lidar_distances, agent.x, agent.y, agent.theta)
                    
                    #-----------------ESTIMATION-----------------
                    # TODO: Implement state estimation here
                    
                    #-----------------MOTION PLANNING-----------------                
                    # TODO: Implement motion planning here

                    #-----------------MOTION CONTROL-----------------
                    # Update the agent's position
                    #agent.move(left_speed, right_speed, dt)
                    # move the agent randomly
                    agent.move(np.random.uniform(-agent.max_v, agent.max_v), np.random.uniform(-agent.max_v, agent.max_v), dt)
                    #-----------------PLOTS-----------------
                    # Plotting the path for the agent
                    ax.plot(agent.target_points[0], agent.target_points[1], color='C' + str(idx), linewidth=1, alpha=0.5)
                    # Plotting the current position of the agent
                    ax.plot(agent.x, agent.y, color='C' + str(idx), alpha=1, marker='o', markersize=3)
                    # Plot the agent's orientation using a line
                    ax.plot([agent.x, agent.x + 1 * np.cos(agent.theta)], [agent.y, agent.y + 1 * np.sin(agent.theta)], color='r', alpha=1, linewidth=1)
                    # Plotting the estimated position of the agent with UWB
                    ax.plot(agent.estim_pos_uwb[0], agent.estim_pos_uwb[1], color='C' + str(idx), alpha=1, marker='o', markersize=3)
                    # Plotting the estimated position of the agent with encoders
                    ax.plot(agent.estim_pos_encoders[0], agent.estim_pos_encoders[1], color='C' + str(idx), alpha=1, marker='o', markersize=3)
                    # Plot LiDAR points around the agent's position
                    lidar_x_coords, lidar_y_coords = zip(*lidar_cartesian_coords)
                    ax.scatter(lidar_x_coords, lidar_y_coords, color='C' + str(idx), marker='o', alpha=0.5, s=0.5)

                    # If the agent reached the current target, remove this target from its path
                    if np.linalg.norm(np.array([agent.x, agent.y]) - np.array(target)) <= 0.1:
                        agent.target_points[0].pop(0)
                        agent.target_points[1].pop(0)

            # If window is closed, then stop the simulation
            if not plt.fignum_exists(ax.get_figure().number):
                return

            plt.pause(dt)

    def plotRoom(self, ax):
        ax.set_title('Room')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        ax.plot(self.room.vertex_x, self.room.vertex_y, 'k', linewidth=1)
        for obstacle in self.room.obstacles:
            ax.plot(obstacle.vertex_x, obstacle.vertex_y, 'k', linewidth=1)
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
        ax.plot(self.room.vertex_x, self.room.vertex_y, color="k", linewidth=1, label="Room")