import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Polygon
import imageio
import io
import cv2
from copy import deepcopy
from Classes.Shapes import Room, Obstacle
from Classes.Agent import Agent
from Classes.Sensors import UWBAnchor, Encoder, Gyroscope, Accelerometer, Magnetometer, DepthSensor
from Classes.VoronoiHandler import VoronoiHandler
from Classes.RobotAssigner import RobotAssigner

class Environment:
    """
    A class representing the environment of the distributed system project.
    
    Attributes:
    - filename (str): the path of the CSV file containing the environment information.
    - agents (list): a list of Agent objects representing the agents in the environment.
    - room (Room): a Room object representing the room in the environment.
    - anchors (list): a list of UWBAnchor objects representing the anchors in the environment.
    - anchors_distances (list): a list of distances between the anchors in the environment.
    - shapes_coord (list): a list of coordinates of the shapes in the environment.
    - ideal_map (list): a list of lines representing the ideal map of the environment.
    - vh (VoronoiHandler): a VoronoiHandler object representing the Voronoi tessellation of the environment.
    - ra (RobotAssigner): a RobotAssigner object representing the robot assigner of the environment.
    - region_paths (list): a list of paths representing the regions in the environment.
    - targetsTotal (int): the total number of targets in the environment.
    - targetsReached (int): the number of targets reached by the agents in the environment.
    - lastTargetGroups (int): the number of groups of agents that have the last target.
    - lastTargetCoords (list): a list of coordinates of the last targets of the agents in the environment.
    - available_agent_index (int): the index of the next available agent in the environment.
    - available_agents (list): a list of available agents in the environment.
    - fig (Figure): a Figure object representing the figure of the environment.
    - axes (Axes): an Axes object representing the axes of the environment.
    - limit_axes (list): a list of the limits of the axes of the environment.
    - axes_offset (int): the offset of the axes of the environment.
    - frames (list): a list of frames representing the video frames of the environment.
    - dt (float): the time step of the environment.
    """
    def __init__(self, filename):
        self.filename = filename
        self.agents = []
        self.room = None
        self.anchors = []
        self.anchors_distances = []
        self.shapes_coord = []
        # create variable to store ideal map of the environment
        self.ideal_map = []
        # create a VoronoiHandler object
        self.vh = None
        # create a RobotAssigner object
        self.ra = None
        # create variables to save regions and paths
        self.region_paths = []

        # create variables to save the number of targets and the total number of targets
        self.targetsTotal = 0
        self.targetsReached = 0
        # split the agents in groups of 2 or 1
        self.lastTargetGroups = None
        self.lastTargetCoords = []
        # create variable to count the number of final targets reached
        self.available_agent_index = 0
        self.available_agents = []

        # create handles for the figure and axes
        self.fig, self.axes = None, None
        self.limit_axes = None
        self.axes_offset = 1

        # save video frames
        self.frames = []
        self.dt = None
    
        self.importCSV()
        
    def importCSV(self):
        """
        Imports the environment information from a CSV file.
        """
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
                
                # get min and max values of the room for limiting the axes
                x_max = np.max(self.shapes_coord[i][0])
                y_max = np.max(self.shapes_coord[i][1])
                x_min = np.min(self.shapes_coord[i][0])
                y_min = np.min(self.shapes_coord[i][1])
                self.limit_axes = [x_min, x_max, y_min, y_max]

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
        """
        Adds a UWBAnchor object to the environment.
        
        Parameters:
        - x (float): the x-coordinate of the anchor.
        - y (float): the y-coordinate of the anchor.
        
        Returns:
        - anchor (UWBAnchor): the UWBAnchor object added to the environment.
        """
        anchor = UWBAnchor(x, y, noise_std_dev=0.1)
        self.anchors.append(anchor)
        return anchor
    
    def createVoronoiTessellation(self, num_points=300):
        """
        Creates a Voronoi tessellation of the environment.
        
        Parameters:
        - num_points (int): the number of points to generate the Voronoi tessellation.
        """
        self.vh = VoronoiHandler([self.room.vertex_x, self.room.vertex_y])
        self.vh.generate_voronoi_within_room(num_points)
    
    def assignRobots(self):
        """
        Assigns the robots to the regions in the environment.
        """
        self.ra = RobotAssigner(self.vh.vor, [self.room.vertex_x, self.room.vertex_y], len(self.agents))
        self.ra.divide_areas_using_kmeans()
        self.region_paths = self.ra.compute_tsp_paths()
        # assign the paths to the agents
        for i in range(len(self.agents)):
            self.agents[i].target_points = [self.region_paths[i][0], self.region_paths[i][1]]
            self.agents[i].ideal_trajectory = deepcopy(self.agents[i].target_points)
            self.targetsTotal += len(self.agents[i].target_points[0])

    def createAgents(self, num_agents):
        """
        Creates the agents in the environment.
        
        Parameters:
        - num_agents (int): the number of agents to create.
        """
        if num_agents > 1:
            for i in range(num_agents-1):
                self.agents.append(Agent(self.agents[0].vertex_x, self.agents[0].vertex_y))
            self.lastTargetGroups = np.ceil(float(num_agents)/2)
    
    def initializeAgentSensors(self, sensors):
        """
        Initializes the sensors of the agents in the environment.
        
        Parameters:
        - sensors (list): a list of strings representing the sensors to initialize.
        """
        for agent in self.agents:
            sensor_list = []
            if 'Encoders' in sensors:
                left_encoder = Encoder(agent.x - agent.wheels_distance/2, agent.y, "EncoderLeft", 720, noise_std_dev=0.2)
                right_encoder = Encoder(agent.x + agent.wheels_distance/2, agent.y, "EncoderRight", 720, noise_std_dev=0.2)
                sensor_list.append(left_encoder)
                sensor_list.append(right_encoder)
            if 'Gyroscope' in sensors:
                gyroscope = Gyroscope(agent.x, agent.y, noise_std_dev=0)
                sensor_list.append(gyroscope)
            if 'Accelerometer' in sensors:
                accelerometer = Accelerometer(agent.x, agent.y, noise_std_dev=0)
                sensor_list.append(accelerometer)
            if 'Magnetometer' in sensors:
                magnetometer = Magnetometer(agent.x, agent.y, noise_std_dev=0.1)
                sensor_list.append(magnetometer)
            if 'Lidar' in sensors:
                lidar = DepthSensor(agent.x, agent.y, "Lidar", 5, 90, 360, noise_std_dev=0.2)
                sensor_list.append(lidar)
            if 'StereoCamera' in sensors:
                stereo_camera = DepthSensor(agent.x, agent.y, "StereoCamera", 3, 240, 120, noise_std_dev=0)
                sensor_list.append(stereo_camera)
            agent.initialize_sensors(sensor_list)

    def addSensorNoise(self, agent, noise=0.1, random=False):
        if random:
            for sensor in agent.sensors.values():
                sensor.noise_std_dev = np.random.uniform(0, 0.1)
        else:
            for sensor in agent.sensors.values():
                sensor.noise_std_dev = noise

    def multilaterationUWB(self, agent, distances):
        result = minimize(self.cost_function, [agent.x, agent.y], args=(self.anchors, distances, np.ones(len(distances))), method='Nelder-Mead')
        return np.array([result.x[0], result.x[1]])
    
    def cost_function(self, initial_guess, anchors, distances, weights):
        eq = []
        for i in range(len(distances)):
            eq.append((np.sqrt((anchors[i].x - initial_guess[0])**2 + (anchors[i].y - initial_guess[1])**2) - distances[i])*weights[i])
        return np.sum(np.array(eq)**2)

    def simulate(self, ax, dt=0.1, saveVideo=False, videoName='Simulation', videoSpeed=1.0):
        
        self.dt = dt
        # Initialize each agent's position to the starting point of its path
        for agent in self.agents:
            agent.x, agent.y = agent.target_points[0][0], agent.target_points[1][0]
            # set the agent's orientation to the direction of the first target point
            agent.theta = np.arctan2(agent.target_points[1][1] - agent.target_points[1][0], agent.target_points[0][1] - agent.target_points[0][0])
            # add a different noise to each agent
            # to keep the original noise comment addSensorNoise(...)
            self.addSensorNoise(agent, noise=0.1, random=True)
            # EKF initialization
            agent.initialize_ekf(np.array([agent.x, agent.y, agent.theta]))
            # state is [x, y, theta], control is [delta_s, delta_theta]
            P_val = np.array([self.anchors[0].noise_std_dev, self.anchors[0].noise_std_dev, agent.sensors["Magnetometer"].noise_std_dev])
            Q_val = np.array([agent.sensors["EncoderLeft"].noise_std_dev, agent.sensors["EncoderRight"].noise_std_dev, agent.sensors["Magnetometer"].noise_std_dev])
            R_val = np.array([agent.sensors["EncoderLeft"].noise_std_dev, agent.sensors["EncoderRight"].noise_std_dev, agent.sensors["Magnetometer"].noise_std_dev])
            agent.initialize_ekf_matrices(P_val, Q_val, R_val)

        while self.targetsReached < self.targetsTotal:
            # Clear the axes and redraw the environment of the first plot
            ax.cla()
            self.plotSimEnv(ax)
            for idx, agent in enumerate(self.agents):
                if len(agent.target_points[0]) > 1:
                    target = [agent.target_points[0][1], agent.target_points[1][1]]
                    
                    #-----------------MOTION CONTROL-----------------
                    # Update the agent's position with proportional control
                    agent.move(target[0], target[1], agent.max_v, dt)
                    # EKF prediction
                    agent.agentEKF.predict([agent.delta_s, agent.delta_theta])
                    
                    #-----------------SENSORS-----------------
                    # UWB MULTILATERATION
                    self.anchors_distances = []
                    for anchor in self.anchors:
                        anchor.update(agent.x, agent.y)
                        self.anchors_distances.append(anchor.get_data())
                    # Update agent's position using multilateration
                    agent.pos_uwb = self.multilaterationUWB(agent, self.anchors_distances)

                    # ENCODERS
                    # Update encoder data for left and right wheels
                    agent.sensors["EncoderLeft"].update(agent.wheel_radius, agent.v - agent.omega * agent.wheels_distance / 2, dt)
                    agent.sensors["EncoderRight"].update(agent.wheel_radius, agent.v + agent.omega * agent.wheels_distance / 2, dt)
                    # Estimate linear and angular displacement using encoders
                    delta_theta_left = agent.get_sensor_data("EncoderLeft") / agent.sensors["EncoderLeft"].ticks_per_revolution * 2 * np.pi
                    delta_theta_right = agent.get_sensor_data("EncoderRight") / agent.sensors["EncoderRight"].ticks_per_revolution * 2 * np.pi
                    delta_theta = 0.5 * (delta_theta_right + delta_theta_left)
                    delta_s = agent.wheel_radius * delta_theta
                    agent.delta_theta_encoders = delta_theta
                    agent.delta_s_encoders = delta_s
                    
                    # MAGNETOMETER
                    # Update magnetometer data
                    agent.sensors["Magnetometer"].update(agent.theta)
                    # Estimate the orientation of the agent
                    agent.theta_mag = agent.sensors["Magnetometer"].get_data()

                    # EKF update
                    # Update using UWB and Magnetometer
                    agent.agentEKF.update([agent.pos_uwb[0], agent.pos_uwb[1], agent.theta_mag])
                    # Update the agent's position
                    agent.x, agent.y, agent.theta = agent.agentEKF.ekf.x_post

                    # LIDAR
                    # Update lidar data
                    agent.sensors["Lidar"].update(agent.x, agent.y, agent.theta, self.ideal_map)
                    # Convert LiDAR distances to cartesian coordinates
                    lidar_cartesian = agent.sensors["Lidar"].convert_to_cartesian(agent.x, agent.y, agent.theta)
                    # Update the scanned map
                    for i in agent.sensors["Lidar"].obstacles_idx:
                        agent.scanned_map.append(lidar_cartesian[i])

                    # STEREO CAMERA
                    # Update stereo camera data
                    agent.sensors["StereoCamera"].update(agent.x, agent.y, agent.theta, self.ideal_map)
                    # Convert stereo camera distances to cartesian coordinates
                    stereo_camera_cartesian = agent.sensors["StereoCamera"].convert_to_cartesian(agent.x, agent.y, agent.theta)
                    # Update the scanned map
                    for i in agent.sensors["StereoCamera"].obstacles_idx:
                        agent.scanned_map.append(stereo_camera_cartesian[i])

                    #-----------------PLOTS-----------------
                    # Plotting the path for the agent
                    ax.plot(agent.target_points[0], agent.target_points[1], color='C' + str(idx), linewidth=1, alpha=0.1)
                    # Plotting the current position of the agent
                    ax.plot(agent.x, agent.y, color='C' + str(idx), alpha=1, marker='o', markersize=3)
                    # Plot the agent's orientation using a line
                    ax.plot([agent.x, agent.x + 1 * np.cos(agent.theta)], [agent.y, agent.y + 1 * np.sin(agent.theta)], color='r', alpha=1, linewidth=1)

                    # Plot LiDAR points around the agent's position
                    lidar_x_coords, lidar_y_coords = zip(*lidar_cartesian)
                    ax.scatter(lidar_x_coords, lidar_y_coords, color='C' + str(idx), marker='o', alpha=0.5, s=0.5)
                    # Plot stereo camera points around the agent's position
                    stereo_camera_x_coords, stereo_camera_y_coords = zip(*stereo_camera_cartesian)
                    ax.scatter(stereo_camera_x_coords, stereo_camera_y_coords, color='C' + str(idx), marker='o', alpha=0.5, s=0.5)
                    # Plot the scanned map
                    if len(agent.scanned_map) > 0:
                        scanned_map_x_coords, scanned_map_y_coords = zip(*agent.scanned_map)
                        ax.scatter(scanned_map_x_coords, scanned_map_y_coords, color='C' + str(idx), marker='o', alpha=1, s=0.5)
                    # If the agent is closed to the current target, remove this target from its path
                    if np.linalg.norm(np.array([agent.x, agent.y]) - np.array(target)) <= 2 and len(agent.target_points[0]) > 1:
                        agent.target_points[0].pop(0)
                        agent.target_points[1].pop(0)
                        self.targetsReached += 1

                    agent.actual_trajectory.append([agent.x, agent.y])
                else:
                    if not agent.reached_final_target:
                        if len(self.available_agents) < self.lastTargetGroups:
                            self.available_agents.append(agent)
                            
                        else:
                            # add position of first available agent as last target
                            agent.target_points[0].append(self.available_agents[self.available_agent_index].x)
                            agent.target_points[1].append(self.available_agents[self.available_agent_index].y)
                            self.available_agent_index += 1
                        
                        agent.reached_final_target = True
                    
                    # if all agents have reached the last target, stop the simulation
                    if self.available_agent_index == int(self.lastTargetGroups):
                        if saveVideo:
                            self.save_video(videoName, videoSpeed)
                        return

                    # plot the scanned map
                    if len(agent.scanned_map) > 0:
                        scanned_map_x_coords, scanned_map_y_coords = zip(*agent.scanned_map)
                        ax.scatter(scanned_map_x_coords, scanned_map_y_coords, color='C' + str(idx), marker='o', alpha=1, s=0.5)
                    # plot the agent's position
                    ax.plot(agent.x, agent.y, color='C' + str(idx), alpha=1, marker='o', markersize=3)

            # If window is closed, then stop the simulation
            if not plt.fignum_exists(ax.get_figure().number):
                return

            plt.pause(dt)

            # Save the current frame
            if saveVideo:
                img = Environment.get_img_from_fig(ax.get_figure())
                self.frames.append(img)
        self.save_video(videoName, videoSpeed)
    
    @staticmethod
    def get_img_from_fig(fig, dpi=100):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def save_video(self, video_name='simulation_video', simSpeed=1.0):

        real_time_fps = 1 / self.dt
        adjusted_fps = real_time_fps * simSpeed
        # get current directory
        current_dir = os.getcwd()
        # chang
        # Create a temporary directory to store the frames if it doesn't exist
        if not os.path.exists(current_dir + "\\temp_frames"):
            os.mkdir(current_dir + "\\temp_frames")
        # save the frames in the temporary directory
        for idx in range(len(self.frames)):
            plt.imsave(f"{current_dir}\\temp_frames\\frame_{idx:04}.png", self.frames[idx])
        
        # use imageio to create the video and save it in the current directory
        for idx in range(len(self.frames)):
            self.frames[idx] = imageio.imread(f"{current_dir}\\temp_frames\\frame_{idx:04}.png")
        video_name = f"{current_dir}\\{video_name}.mp4"
        imageio.mimsave(video_name, self.frames, fps=adjusted_fps)
        
        # print(f"Video saved as {video_name}")

        # Clean up temporary frames
        for idx in range(len(self.frames)):
            os.remove(f"{current_dir}\\temp_frames\\frame_{idx:04}.png")

        os.rmdir(current_dir + "\\temp_frames")

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
        
        ax.set_xlim(self.limit_axes[0]-self.axes_offset, self.limit_axes[1]+self.axes_offset)
        ax.set_ylim(self.limit_axes[2]-self.axes_offset, self.limit_axes[3]+self.axes_offset)
    
    def plotVoronoiTessellation(self, ax):
        ax.set_title('Voronoi Tessellation')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        for line in self.vh.clipped_ridges:
            x, y = line.xy
            ax.plot(x, y, color="black", linewidth=0.5)

        ax.set_xlim(self.limit_axes[0]-self.axes_offset, self.limit_axes[1]+self.axes_offset)
        ax.set_ylim(self.limit_axes[2]-self.axes_offset, self.limit_axes[3]+self.axes_offset)
        
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
        
        ax.set_xlim(self.limit_axes[0]-self.axes_offset, self.limit_axes[1]+self.axes_offset)
        ax.set_ylim(self.limit_axes[2]-self.axes_offset, self.limit_axes[3]+self.axes_offset)
        
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
        
        ax.set_xlim(self.limit_axes[0]-self.axes_offset, self.limit_axes[1]+self.axes_offset)
        ax.set_ylim(self.limit_axes[2]-self.axes_offset, self.limit_axes[3]+self.axes_offset)
    
    def plotSimEnv(self, ax):
        ax.set_title('Simulation Environment')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='dotted')
        
        ax.set_xlim(self.limit_axes[0]-self.axes_offset, self.limit_axes[1]+self.axes_offset)
        ax.set_ylim(self.limit_axes[2]-self.axes_offset, self.limit_axes[3]+self.axes_offset)