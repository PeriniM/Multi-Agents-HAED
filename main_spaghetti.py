import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Polygon

def discretize_element(x, y, resolution):
    # This function discretizes a 2D element into a series of points equally spaced on both axes
    
    # Add the first point to the end of the list to close the polygon
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    x_ob = []
    y_ob = []

    # Loop through each side of the element
    for j in range(len(x)-1):
        # Calculate the number of steps needed to discretize the side
        length_side = np.sqrt((x[j]-x[j+1])**2 + (y[j]-y[j+1])**2)
        num_steps = int(np.ceil(length_side / resolution))
        # Add the discretized points to the vector
        x_ob.extend(np.linspace(x[j], x[j+1], num_steps))
        y_ob.extend(np.linspace(y[j], y[j+1], num_steps))
    
    return x_ob, y_ob

def cost_function(initial_guess, anchors, distances, weights):
    eq = []
    for i in range(len(distances)):
        eq.append(weights[i] * (initial_guess[0] - anchors[i][0])**2 + (initial_guess[1] - anchors[i][1])**2 - distances[i]**2)
    return np.sum(np.array(eq)**2)


# Function to calculate the intersection point between two segments
def segment_intersection(p1, p2, p3, p4):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2
    
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    denominator = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return np.array([
            ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denominator,
            ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denominator
        ])
    
    if o1 == 0 and on_segment(p1, p3, p2):
        return p3
    if o2 == 0 and on_segment(p1, p4, p2):
        return p4
    if o3 == 0 and on_segment(p3, p1, p4):
        return p1
    if o4 == 0 and on_segment(p3, p2, p4):
        return p2
    
    return None

# Lidar scan using ray casting algorithm
def lidar_scan(estim_pos, estim_orientation, real_map_vert, resolution=100, max_range=20):
    '''
    estim_pos: estimated position of the robot [x_CoM, y_CoM]
    estim_orientation: estimated orientation of the robot [angle]
    real_map_vert: vertices of the real map
    resolution: number of rays to cast
    max_range: maximum range of the lidar
    '''
    # Initialize the lidar scan
    raycast_distances = np.zeros(resolution)
    raycast_angles = np.linspace(0, 2 * np.pi, resolution)
    isIntersected = np.zeros(resolution)
    # Initialize the intersection point to None
    intersection_point = None

    # Loop through each ray
    for i in range(resolution):
        # Calculate the angle of the ray
        angle = estim_orientation + i * 2 * np.pi / resolution
        # Initialize the minimum distance
        min_distance = max_range
        # Loop through each side of the real map
        for j in range(len(real_map_vert)):
            # Loop through each vertex of the side
            for k in range(len(real_map_vert[j][0]) - 1):
                # Calculate the intersection point between the ray and the side
                intersection_candidate = segment_intersection(estim_pos, estim_pos + np.array([np.cos(angle), np.sin(angle)]) * max_range, np.array([real_map_vert[j][0][k], real_map_vert[j][1][k]]), np.array([real_map_vert[j][0][k + 1], real_map_vert[j][1][k + 1]]))
                # Check if the intersection point is not None
                if intersection_candidate is not None:
                    # Calculate the distance between the robot and the intersection point
                    distance = np.linalg.norm(intersection_candidate - estim_pos)
                    # Check if the distance is smaller than the minimum distance
                    if distance < min_distance:
                        # Update the minimum distance
                        min_distance = distance
                        intersection_point = intersection_candidate
        # Update the lidar scan
        raycast_distances[i] = min_distance if intersection_point is not None else max_range
        if raycast_distances[i] < max_range:
            isIntersected[i] = 1
    return raycast_distances, raycast_angles + estim_orientation, isIntersected
        

# Import the CSV containing the shapes
df = pd.read_csv('Rooms/dungeon.csv', header=None, skiprows=1, na_values=['NA', 'na'])
shape = []
x_max = -1e5
y_max = -1e5
x_min = 1e5
y_min = 1e5

# Build each shape with its info
for i in range(len(df)):
    x = np.fromstring(df.iloc[i, 1], sep=' ')
    y = np.fromstring(df.iloc[i, 2], sep=' ') * -1
    element_type = df.iloc[i, 4]
    shape.append([x, y, element_type])

    # Update min and max for normalization
    x_max = max(x_max, np.max(x))
    y_max = max(y_max, np.max(y))
    x_min = min(x_min, np.min(x))
    y_min = min(y_min, np.min(y))

# Normalize coordinates in both axes
width_room = abs(x_max - x_min)
height_room = abs(y_max - y_min)
aspect_ratio = width_room / height_room

x_scale = [0, width_room]
y_scale = [0, height_room]

for j in range(len(shape[0])-1):
    for i in range(len(shape)):
        if j == len(shape[0]) - 2:
            shape[i][j] = (shape[i][j] - y_min) * (y_scale[1] - y_scale[0]) / (y_max - y_min) + y_scale[0]
        else:
            shape[i][j] = (shape[i][j] - x_min) * (x_scale[1] - x_scale[0]) / (x_max - x_min) + x_scale[0]

# Plot normalized shape
# plt.figure(1)
# for i in range(len(shape)):
#     plt.plot(np.append(shape[i][0], shape[i][0][0]), np.append(shape[i][1], shape[i][1][0]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(True, which='both', linestyle='dotted')
# plt.show()


# Separate the elements into their respective lists
agent_vert = []
obstacle_vert = []
room_vert = []

for i in range(len(shape)):
    if shape[i][2] == 'room':
        room_vert.append([shape[i][0], shape[i][1]])
    elif shape[i][2] == 'obstacle':
        obstacle_vert.append([shape[i][0], shape[i][1]])
    elif shape[i][2] == 'agent':
        agent_vert.append([shape[i][0], shape[i][1], [np.mean(shape[i][0]), np.mean(shape[i][1])]])
# array containing points of all the elements except the agent
map_real_vert = np.array(room_vert + obstacle_vert, dtype=object)
# Discretize resolution
resolution = 0.5

# Discretize the obstacles
for i in range(len(obstacle_vert)):
    obstacle_vert[i].extend(discretize_element(obstacle_vert[i][0], obstacle_vert[i][1], resolution))

# Discretize the room
for i in range(len(room_vert)):
    room_vert[i].extend(discretize_element(room_vert[i][0], room_vert[i][1], resolution))

# Plot discretized shape
# plt.figure(2)
# for i in range(len(obstacle_vert)):
#     plt.plot(obstacle_vert[i][2], obstacle_vert[i][3], 'o', markersize=2)
# for i in range(len(room_vert)):
#     plt.plot(room_vert[i][2], room_vert[i][3], 'o', markersize=2)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(True, which='both', linestyle='dotted')
# plt.show()

# Add the UWB anchors at the corners of the room
anchors = []
for i in range(len(room_vert)):
    for j in range(len(room_vert[i][0])):
        anchors.append([room_vert[i][0][j], room_vert[i][1][j]])

# Plot the anchors
# plt.figure(2)
# for i in range(len(anchors)):
#     plt.plot(anchors[i][0], anchors[i][1], 'o', markersize=8, markerfacecolor='r', markeredgecolor='r')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(True, which='both', linestyle='dotted')
# plt.show()

# Add unicycle dynamics to the agents
x_CoM = agent_vert[0][2][0]
y_CoM = agent_vert[0][2][1]
theta = np.random.rand() * 2 * np.pi

# Define the maximum speed and angular velocity of the agent
max_speed = 100  # meters per second
max_angular_velocity = np.pi*10  # radians per second

# Define the time step for the simulation
dt = 0.01  # seconds

# Translate the center of mass to the origin
agent_vertices = np.array(agent_vert[0][:2]).T - [x_CoM, y_CoM]

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

# set the aspect ratio and grid properties for the first subplot
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, which='both', linestyle='dotted')

# set the aspect ratio and grid properties for the second subplot
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, which='both', linestyle='dotted')

# Add the room and obstacles to the plot
for i in range(len(map_real_vert)):
    # plot the real map closing the polygon
    ax1.plot(np.append(map_real_vert[i][0], map_real_vert[i][0][0]), np.append(map_real_vert[i][1], map_real_vert[i][1][0]), 'k', linewidth=1)
    # for each element of the map, append the last point to close the polygon
    map_real_vert[i][0] = np.append(map_real_vert[i][0], map_real_vert[i][0][0])
    map_real_vert[i][1] = np.append(map_real_vert[i][1], map_real_vert[i][1][0]) 
    
# Add the anchors to the plot
for i in range(len(anchors)):
    ax1.plot(anchors[i][0], anchors[i][1], 'o', markersize=2, markerfacecolor='r', markeredgecolor='r')
    ax2.plot(anchors[i][0], anchors[i][1], 'o', markersize=2, markerfacecolor='r', markeredgecolor='r')

# Plot the agent as a Polygon
h_agent = Polygon(agent_vertices, edgecolor='blue', facecolor='blue', alpha=0.7)
ax1.add_patch(h_agent)

# Plot the center of mass as a red dot
h_CoM, = ax1.plot(x_CoM, y_CoM, 'o', markersize=3, markerfacecolor='r', markeredgecolor='r')

# Plot the estimated center of mass connections to anchors
h_CoM_estim = [ax2.plot([], [], 'r')[0] for _ in anchors]
h_CoM_estim_point, = ax2.plot([], [], 'ro')
# ones vector for the weights of the cost function
estimate_weights = np.ones(len(anchors))

# Initialize a magnetometer sensor with 360 degrees of resolution
mag_resolution = 360
mag = np.linspace(0, 2*np.pi, mag_resolution)

north_pole = np.array([0, 1])  # Unit vector pointing north

# Plot the real and sensor readings of the robot's orientation
h_north_pole, = ax2.plot([], [], 'purple')
# plot the magnetometer sensor readings with a line
h_mag_orientation, = ax2.plot([], [], 'b')

# plot lidar scan in gray with dots and thin lines
h_lidar_scan, = ax2.plot([], [], 'k.', markersize=1, linewidth=0.5)

# save old robot position
real_robot_position = []
estimate_robot_position = []

# plot the real robot position
# h_real_robot_position, = ax1.plot([], [], 'go', markersize=1)
# plot the estimated robot position
h_estimate_robot_position, = ax2.plot([], [], 'go', markersize=1)

occupancy_points = []
h_occupancy_points, = ax2.plot([], [], 'ko', markersize=1)

# ----- SIMULATION ----- #
counter = 0
while True:
    # Generate random velocity and angular velocity for the robot
    speed = np.random.rand() * max_speed
    angular_velocity = (np.random.rand() - 0.5) * 2 * max_angular_velocity

    # Calculate the x and y components of the velocity and orientation
    vx = speed * np.cos(theta)
    vy = speed * np.sin(theta)
    omega = angular_velocity

    # Update the position and orientation of the robot
    x_CoM += vx * dt
    y_CoM += vy * dt
    theta += omega * dt

    # Update the Polygon for the agent's position
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_vertices = np.dot(agent_vertices, rotation_matrix.T)
    rotated_vertices[:, 0] += x_CoM
    rotated_vertices[:, 1] += y_CoM
    h_agent.set_xy(rotated_vertices)

    distance = []
    # Perform trilateration to estimate the position of the robot's center of mass
    for i in range(len(anchors)):
        # Calculate the distance between the robot and the anchor
        distance.append(np.linalg.norm(np.array([x_CoM, y_CoM]) - np.array(anchors[i])))  # Convert lists to NumPy arrays

    # Minimize the objective function using minimize_scalar
    result = minimize(lambda x: cost_function(x, anchors, distance, estimate_weights), x0=[0, 0], method='Nelder-Mead', tol=1e-6, options={'maxiter': 1000})
    CoM_estim = result.x

    # Update the graphics object for the robot's real orientation
    real_robot_orientation = [np.cos(theta), np.sin(theta)]
    # calculate the sensor reading of the robot's orientation respecting the real north
    mag_orientation = np.arccos(np.dot(real_robot_orientation, north_pole) / (np.linalg.norm(real_robot_orientation) * np.linalg.norm(north_pole)))
    # get the closest value to the sensor reading from the magnetometer
    mag_orientation = mag[np.argmin(np.abs(mag - mag_orientation))]
    # check if orientation is in the first or second quadrant
    if real_robot_orientation[0] < 0:
        mag_orientation = 2 * np.pi - mag_orientation

    # Get lidar scan distances and angles
    lidar_scan_distances, lidar_scan_angles, lidar_scan_intersected = lidar_scan(CoM_estim, mag_orientation, map_real_vert)


    # every 10 iterations append the real and estimated robot position
    if counter % 100 == 0:
        real_robot_position.append([x_CoM, y_CoM])
        estimate_robot_position.append(CoM_estim)

    # ----- UPDATE THE GRAPHICS ----- #
    # Plot the lidar scan
    h_lidar_scan.set_data(lidar_scan_distances * np.cos(lidar_scan_angles) + CoM_estim[0], lidar_scan_distances * np.sin(lidar_scan_angles) + CoM_estim[1])
    # append the lidar scan points to the occupancy points only if the ray intersects with an obstacle
    for i in range(len(lidar_scan_intersected)):
        if lidar_scan_intersected[i] == 1:
            # if the point is not already in the occupancy points, append it
            if [lidar_scan_distances[i] * np.cos(lidar_scan_angles[i]) + CoM_estim[0], lidar_scan_distances[i] * np.sin(lidar_scan_angles[i]) + CoM_estim[1]] not in occupancy_points:
                occupancy_points.append([lidar_scan_distances[i] * np.cos(lidar_scan_angles[i]) + CoM_estim[0], lidar_scan_distances[i] * np.sin(lidar_scan_angles[i]) + CoM_estim[1]])
    
    # Plot the occupancy points
    if len(occupancy_points) > 0:
        h_occupancy_points.set_data(np.array(occupancy_points).T[0], np.array(occupancy_points).T[1])
    # ----- UPDATE THE GRAPHICS ----- #

    # Update the graphics object for the robot's center of mass
    h_CoM.set_data([x_CoM], [y_CoM])  # Wrap x_CoM and y_CoM with lists

    # Update the graphics connecting the robot's estimated position to all the anchors
    # for i in range(len(anchors)):
    #     h_CoM_estim[i].set_data([CoM_estim[0], anchors[i][0]], [CoM_estim[1], anchors[i][1]])
    # Update the graphics object for the robot's estimated position
    h_CoM_estim_point.set_data([CoM_estim[0]], [CoM_estim[1]])
    # Update the graphics object for the robot's real north
    h_north_pole.set_data([x_CoM, x_CoM + north_pole[0]*2], [y_CoM, y_CoM + north_pole[1]*2])
    # Update the graphics object for the robot's estimated orientation considering the sensor reading and the real north
    h_mag_orientation.set_data([x_CoM, x_CoM + np.sin(mag_orientation)*3], [y_CoM, y_CoM + np.cos(mag_orientation)*3])

    # Update the graphics object for the robot's real position
    # h_real_robot_position.set_data(np.array(real_robot_position).T[0], np.array(real_robot_position).T[1])
    # Update the graphics object for the robot's estimated position
    h_estimate_robot_position.set_data(np.array(estimate_robot_position).T[0], np.array(estimate_robot_position).T[1])
    # Pause for a short time to simulate real-time behavior
    plt.pause(dt)

