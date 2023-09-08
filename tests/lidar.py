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
    
class VFH:
    def __init__(self, num_bins, threshold):
        self.num_bins = num_bins
        self.threshold = threshold
        self.histogram = np.zeros(num_bins)
        
    def create_histogram(self, lidar):
        bin_width = len(lidar.readings) // self.num_bins
        for i in range(self.num_bins):
            self.histogram[i] = np.min(lidar.readings[i*bin_width:(i+1)*bin_width])
    
    def get_target_direction(self, desired_direction):
        # Threshold the histogram
        blocked_bins = np.where(self.histogram < self.threshold)[0]

        # Check if desired direction is free
        desired_bin = (desired_direction / (2 * np.pi)) * self.num_bins
        if desired_bin not in blocked_bins:
            return desired_direction

        # Otherwise, find the largest gap
        gaps = np.diff(blocked_bins)
        largest_gap_idx = np.argmax(gaps)
        
        # Return the center of the largest gap as the new target direction
        target_bin = (blocked_bins[largest_gap_idx] + blocked_bins[largest_gap_idx + 1]) // 2
        return (target_bin / self.num_bins) * (2 * np.pi)

class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = 1.0
        
    def move(self, direction, dt=1):
        self.x += self.speed * np.cos(direction) * dt
        self.y += self.speed * np.sin(direction) * dt
        self.theta = direction

def plot_robot(robot, ax):
    ax.scatter([robot.x], [robot.y], color='b', marker='o')
    dx = 1.0 * np.cos(robot.theta)
    dy = 1.0 * np.sin(robot.theta)
    ax.arrow(robot.x, robot.y, dx, dy, head_width=0.5, head_length=0.5, fc='blue', ec='blue')

def plot_lidar_readings(lidar, robot, ax):
    for i, distance in enumerate(lidar.readings):
        angle = lidar.beam_angles[i] + robot.theta
        x_end = robot.x + distance * np.cos(angle)
        y_end = robot.y + distance * np.sin(angle)
        ax.plot([robot.x, x_end], [robot.y, y_end], color='r', alpha=0.5)

def compute_histogram(lidar):
    hist_bins = np.linspace(0, 360, lidar.num_beams + 1)  # Create bins for 360 degrees.
    hist_vals, _ = np.histogram(lidar.beam_angles * (180 / np.pi), bins=hist_bins, weights=1 / lidar.readings)
    return hist_vals, hist_bins

def main():
    # Define the environment as a list of line segments
    ideal_map = [
        [0, 0, 12, 0],  # bottom
        [10, 0, 12, 10],  # right
        [10, 10, 0, 10],  # top
        [0, 10, 0, 0],   # left
        [3, 3, 7, 3],    # horizontal obstacle
        [7, 3, 7, 7],    # vertical obstacle
    ]

    robot = Robot(5, 1, np.pi/2)  # Initialize robot position and heading direction
    lidar = LiDAR(robot.x, robot.y, "LiDAR", 15, 360, 360)
    vfh = VFH(num_bins=36, threshold=2.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot for several time steps
    for _ in range(10):
        # Clear previous data
        ax1.cla()
        ax2.cla()
        
        # Draw environment
        for segment in ideal_map:
            ax1.plot([segment[0], segment[2]], [segment[1], segment[3]], 'k-')

        # Update LiDAR
        lidar.update(robot.x, robot.y, robot.theta, ideal_map)
        # Plot LiDAR readings
        plot_lidar_readings(lidar, robot, ax1)

        # Update and plot VFH histogram
        vfh.create_histogram(lidar)
        ax2.bar(np.arange(vfh.num_bins), vfh.histogram, width=1.0)

        # Update robot's direction and move
        target_direction = vfh.get_target_direction(robot.theta)
        robot.move(target_direction)

        # Plot robot
        plot_robot(robot, ax1)

        ax1.set_xlim(-5, 15)
        ax1.set_ylim(-5, 15)
        ax2.set_ylim(0, lidar.max_range+1)
        plt.pause(0.5)  # Wait for a short duration before plotting the next frame

    plt.show()

if __name__ == "__main__":
    main()

