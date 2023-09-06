import numpy as np
import matplotlib.pyplot as plt

# Assuming you have defined the Sensor class somewhere
class Sensor:
    def __init__(self, x, y, sensor_type, noise_std_dev):
        self.x = x
        self.y = y
        self.sensor_type = sensor_type
        self.noise_std_dev = noise_std_dev

class LiDAR(Sensor):
    def __init__(self, x, y, sensor_type, max_range, num_beams, fov_degrees, noise_std_dev=0.1):
        super().__init__(x, y, sensor_type, noise_std_dev)
        self.max_range = max_range
        self.num_beams = num_beams
        self.fov = np.deg2rad(fov_degrees)  # Convert FOV to radians
        self.beam_angles = np.linspace(-self.fov/2, self.fov/2, num_beams)
        self.readings = np.zeros(num_beams)

    def update(self, agent_x, agent_y, agent_theta, ideal_map):
        for i, angle in enumerate(self.beam_angles):
            dx = self.max_range * np.cos(agent_theta + angle)
            dy = self.max_range * np.sin(agent_theta + angle)
            
            # Determine the end point of the beam (if it went the max distance without an obstacle)
            end_x = agent_x + dx
            end_y = agent_y + dy
            
            # Find the closest intersection with the obstacles
            min_distance = self._check_intersection(agent_x, agent_y, end_x, end_y, ideal_map)
            if min_distance is None:  # No obstacle detected within max_range
                min_distance = self.max_range
            
            # Apply noise if specified
            noise = np.random.normal(0, self.noise_std_dev) if self.noise_std_dev > 0 else 0
            self.readings[i] = min_distance + noise

    def _check_intersection(self, x1, y1, x2, y2, ideal_map):
        min_distance = None
        for segment in ideal_map:
            x3, y3, x4, y4 = segment

            # Compute determinants
            detA = (x1 - x3) * (y4 - y3) - (y1 - y3) * (x4 - x3)
            detB = (x2 - x3) * (y4 - y3) - (y2 - y3) * (x4 - x3)
            detC = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
            detD = (x4 - x1) * (y2 - y1) - (y4 - y1) * (x2 - x1)

            if (detA * detB < 0) and (detC * detD < 0):
                # Lines intersect
                det = (x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3)
                
                # Avoid division by zero
                if det == 0:
                    continue

                # Compute intersection point
                alpha = ((x1 - x3) * (y4 - y3) - (y1 - y3) * (x4 - x3)) / det
                xi = x1 + alpha * (x2 - x1)
                yi = y1 + alpha * (y2 - y1)
                distance = np.sqrt((xi - x1)**2 + (yi - y1)**2)
                
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance

    
    def convert_to_cartesian(self, distances, robot_x, robot_y, robot_theta):
        cartesian_coords = []
        for i, distance in enumerate(distances):
            angle = self.beam_angles[i] + robot_theta
            x = robot_x + distance * np.cos(angle)
            y = robot_y + distance * np.sin(angle)
            cartesian_coords.append((x, y))
        return cartesian_coords

    def get_data(self):
        return self.readings

# Test Script starts here

def plot_lidar_readings(lidar, ax):
    for i, distance in enumerate(lidar.readings):
        angle = lidar.beam_angles[i]
        x = lidar.x + distance * np.cos(angle)
        y = lidar.y + distance * np.sin(angle)
        ax.plot([lidar.x, x], [lidar.y, y], color='r', alpha=0.5)

def main():
    # Define the environment as a list of line segments
    ideal_map = [
        [0, 0, 12, 0],  # bottom
        [10, 0, 12, 10],  # right
        [10, 10, 0, 10],  # top
        [0, 10, 0, 0]   # left
    ]

    lidar = LiDAR(5, 5, "LiDAR", 15, 360, 360)  # Placing the LiDAR in the center of the environment

    # Plot the environment and the LiDAR readings
    fig, ax = plt.subplots()

    # Draw environment
    for segment in ideal_map:
        ax.plot([segment[0], segment[2]], [segment[1], segment[3]], 'k-')

    # Update LiDAR
    lidar.update(lidar.x, lidar.y, 0, ideal_map)

    # Plot LiDAR readings
    plot_lidar_readings(lidar, ax)

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    plt.show()

if __name__ == "__main__":
    main()
