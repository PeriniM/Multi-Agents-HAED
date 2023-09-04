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
        self.magnetic_field = [0, 0]  # Assuming 2D magnetic field [mx, my]

    def update(self, true_magnetic_field):
        self.magnetic_field = [self.add_noise(val) for val in true_magnetic_field]

    def get_data(self):
        return self.magnetic_field