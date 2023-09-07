import numpy as np

class Sensor:
    def __init__(self, x, y, sensor_type, noise_std_dev=0.0):
        self.x = x
        self.y = y
        self.sensor_type = sensor_type
        self.noise_std_dev = noise_std_dev

    def add_noise(self, true_value):
        noise = np.random.normal(0, self.noise_std_dev) if self.noise_std_dev > 0 else 0
        return true_value + noise

class Encoder(Sensor):
    def __init__(self, x, y, sensor_type, ticks_per_revolution, noise_std_dev=0.1, bias=0.0):
        super().__init__(x, y, sensor_type, noise_std_dev)
        self.ticks = 0
        self.ticks_per_revolution = ticks_per_revolution
        self.bias = bias

    def update(self, wheel_radius, speed, dt):
        true_ticks = self.ticks_per_revolution * (speed * dt) / (2 * np.pi * wheel_radius)
        self.ticks += self.add_noise(true_ticks + self.bias)

    def get_data(self):
        return self.ticks
    
class UWBAnchor(Sensor):
    def __init__(self, x, y, noise_std_dev=1.0):
        super().__init__(x, y, 'UWBAnchor', noise_std_dev)
        self.robot_distance = None

    def update(self, robot_x, robot_y):
        true_distance = np.sqrt((self.x - robot_x)**2 + (self.y - robot_y)**2)
        self.robot_distance = self.add_noise(true_distance)

    def get_data(self):
        return self.robot_distance

class Gyroscope(Sensor):
    def __init__(self, x, y, noise_std_dev=0.1):
        super().__init__(x, y, 'Gyroscope', noise_std_dev)
        self.angular_velocity = 0

    def update(self, true_angular_velocity):
        self.angular_velocity = self.add_noise(true_angular_velocity)

    def get_data(self):
        return self.angular_velocity

class Accelerometer(Sensor):
    def __init__(self, x, y, noise_std_dev=0.1):
        super().__init__(x, y, 'Accelerometer', noise_std_dev)
        self.acceleration = [0, 0]  # Assuming 2D acceleration [ax, ay]

    def update(self, true_acceleration):
        self.acceleration = [self.add_noise(val) for val in true_acceleration]

    def get_data(self):
        return self.acceleration

class Magnetometer(Sensor):
    def __init__(self, x, y, noise_std_dev=0.1):
        super().__init__(x, y, 'Magnetometer', noise_std_dev)
        self.orientation = 0

    def update(self, theta):
        self.orientation = self.add_noise(theta)

    def get_data(self):
        return self.orientation
    
class DepthSensor(Sensor):
    def __init__(self, x, y, sensor_type, max_range, num_beams, fov_degrees, noise_std_dev=0.1):
        super().__init__(x, y, sensor_type, noise_std_dev)
        self.max_range = max_range
        self.num_beams = num_beams
        self.fov = np.deg2rad(fov_degrees)  # Convert FOV to radians
        self.beam_angles = np.linspace(-self.fov/2, self.fov/2, num_beams)
        self.readings = np.zeros(num_beams)
        self.obstacles_idx = []

    def update(self, agent_x, agent_y, agent_theta, ideal_map):
        self.obstacles_idx = []
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
            else:
                self.obstacles_idx.append(i)

            # Apply noise if specified
            noise = np.random.normal(0, self.noise_std_dev) if self.noise_std_dev > 0 else 0
            self.readings[i] = min_distance + noise

    def _check_intersection(self, x1, y1, x2, y2, ideal_map):
        min_distance = None
        # Merge all elements into a single list
        merged_segments = []
        for segment_list in ideal_map:
            merged_segments.extend(segment_list)

        for segment in merged_segments:
            x3, y3, x4, y4 = segment

            # Compute determinants
            detA = (x1 - x3) * (y4 - y3) - (y1 - y3) * (x4 - x3)
            detB = (x2 - x3) * (y4 - y3) - (y2 - y3) * (x4 - x3)
            detC = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
            detD = (x4 - x1) * (y2 - y1) - (y4 - y1) * (x2 - x1)

            if (detA * detB < 0).all() and (detC * detD < 0).all():
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
    
    def convert_to_cartesian(self, robot_x, robot_y, robot_theta):
        cartesian_coords = []
        for i, distance in enumerate(self.readings):
            angle = self.beam_angles[i] + robot_theta
            x = robot_x + distance * np.cos(angle)
            y = robot_y + distance * np.sin(angle)
            cartesian_coords.append((x, y))
        return cartesian_coords
    
    def get_data(self):
        return self.readings
    

