import numpy as np
from Classes.Shapes import Shape
from Classes.EKF import EKF

class Agent(Shape):
    """
    A class representing an agent in a distributed system.

    Attributes:
    - vertex_x (list): A list of x-coordinates of the vertices of the agent's shape.
    - vertex_y (list): A list of y-coordinates of the vertices of the agent's shape.
    - x (float): The x-coordinate of the center of the agent's shape.
    - y (float): The y-coordinate of the center of the agent's shape.
    - theta (float): The orientation of the agent's shape in radians.
    - v (float): The linear velocity of the agent.
    - omega (float): The angular velocity of the agent.
    - max_v (float): The maximum linear velocity of the agent.
    - max_omega (float): The maximum angular velocity of the agent.
    - kp (float): The proportional gain of the agent's controller.
    - wheels_distance (float): The distance between the wheels of the agent.
    - wheel_radius (float): The radius of the wheels of the agent.
    - dynamics (str): The type of dynamics of the agent.
    - sensors (dict): A dictionary of the agent's sensors.
    - agentEKF (ExtendedKalmanFilter): The Extended Kalman Filter used for estimation.
    - pos_uwb (np.ndarray): The position of the agent as estimated by UWB.
    - delta_theta_encoders (float): The change in orientation of the agent as estimated by encoders.
    - delta_s_encoders (float): The change in position of the agent as estimated by encoders.
    - theta_mag (float): The orientation of the agent as estimated by magnetometer.
    - delta_s (float): The change in position of the agent.
    - delta_theta (float): The change in orientation of the agent.
    - target_points (np.ndarray): The target points of the path to follow.
    - scanned_map (list): The map scanned by the agent.
    - reached_final_target (bool): Whether the agent has reached the final target.

    Methods:
    - getVertices(): Returns the transformed vertices of the agent's shape.
    - initialize_sensors(sensors): Initializes the agent's sensors.
    - initialize_ekf(initial_state): Initializes the Extended Kalman Filter.
    - initialize_ekf_matrices(P_val, Q_val, R_val): Initializes the matrices of the Extended Kalman Filter.
    - move(target_x, target_y, desired_speed, dt): Moves the agent towards a target point.
    - update(target_x, target_y, desired_speed): Updates the agent's velocities based on the error in position and angle.
    - get_sensor_data(sensor_name): Returns the data from a sensor.
    """

    def __init__(self, vertex_x, vertex_y):
        """
        Initializes an Agent object.

        Args:
        - vertex_x (list): A list of x-coordinates of the vertices of the agent's shape.
        - vertex_y (list): A list of y-coordinates of the vertices of the agent's shape.
        """
        super().__init__(vertex_x, vertex_y, element_type="agent")
        self.vertex_x = vertex_x
        self.vertex_y = vertex_y
        self.x = np.mean(vertex_x)
        self.y = np.mean(vertex_y)
        self.theta = 0
        self.v = 0
        self.omega = 0

        # set max speed and angular velocity
        self.max_v = 5
        self.max_omega = 5
        # set PID constants
        self.kp = 1000

        # set wheels distance as the minimum distance between the middle point and the closest vertex
        self.wheels_distance = np.min(np.sqrt((self.x - vertex_x)**2 + (self.y - vertex_y)**2))
        self.wheel_radius = self.wheels_distance / 2

        self.dynamics = "differential"
        self.sensors = {}

        # Initialize the Extended Kalman Filter for the agent
        self.agentEKF = None

        # Estimation variables
        self.pos_uwb = None
        self.delta_theta_encoders = None
        self.delta_s_encoders = None
        self.theta_mag = None
        self.delta_s = None
        self.delta_theta = None
        
    
        # Iniitialize the target points of the path to follow 
        self.target_points = None
        self.scanned_map = []
        self.reached_final_target = False
        
    def getVertices(self):
        """
        Returns the transformed vertices of the agent's shape.

        Returns:
        - np.ndarray: The transformed vertices of the agent's shape.
        """
        # return transformed vertices
        new_vertex_x = self.x + (self.vertex_x - self.x) * np.cos(self.theta) + (self.vertex_y - self.y) * np.sin(self.theta)
        new_vertex_y = self.y - (self.vertex_x - self.x) * np.sin(self.theta) + (self.vertex_y - self.y) * np.cos(self.theta)
        return np.array([new_vertex_x, new_vertex_y]).T
    
    def initialize_sensors(self, sensors):
        """
        Initializes the agent's sensors.

        Args:
        - sensors (list): A list of Sensor objects.
        """
        for sensor in sensors:
            if sensor.sensor_type == "Encoder":
                self.sensors["Encoder_left"] = sensor
                self.sensors["Encoder_right"] = sensor
            else:
                self.sensors[sensor.sensor_type] = sensor

    def initialize_ekf(self, initial_state):
        """
        Initializes the Extended Kalman Filter.

        Args:
        - initial_state (np.ndarray): The initial state of the filter.
        """
        self.agentEKF = EKF(dim_x=3, dim_z=3, initial_state=initial_state)
    
    def initialize_ekf_matrices(self, P_val, Q_val, R_val):
        """
        Initializes the matrices of the Extended Kalman Filter.

        Args:
        - P_val (float): The initial state covariance.
        - Q_val (float): The process noise.
        - R_val (float): The measurement noise.
        """
        # Initial state covariance. P_val is a 1x3 array
        self.agentEKF.ekf.P = np.diag(P_val**2)
        # Process noise. Q_val is a 1x3 array
        self.agentEKF.ekf.Q = np.diag(Q_val**2)
        # Measurement noise. R_val is a 1x3 array
        self.agentEKF.ekf.R = np.diag(R_val**2)

    def move(self, target_x, target_y, desired_speed, dt):
        """
        Moves the agent towards a target point.

        Args:
        - target_x (float): The x-coordinate of the target point.
        - target_y (float): The y-coordinate of the target point.
        - desired_speed (float): The desired speed of the agent.
        - dt (float): The time step.

        Returns:
        - None
        """
        # Use the proportional controller to update velocities
        self.update(target_x, target_y, desired_speed)

        # Ensure linear and angular velocities are within limits
        self.v = np.clip(self.v, -self.max_v, self.max_v)
        self.omega = np.clip(self.omega, -self.max_omega, self.max_omega)
        
        # Update the agent's pose
        if self.dynamics == "differential":
            # Calculate the new pose
            self.delta_s = self.v * dt
            self.delta_theta = self.omega * dt
            self.x += self.delta_s * np.cos(self.theta + 0.5 * self.delta_theta)
            self.y += self.delta_s * np.sin(self.theta + 0.5 * self.delta_theta)
            self.theta += self.delta_theta
        else:
            raise ValueError("Unknown dynamics type: {}".format(self.dynamics))
    
    def update(self, target_x, target_y, desired_speed):
        """
        Updates the agent's velocities based on the error in position and angle.

        Args:
        - target_x (float): The x-coordinate of the target point.
        - target_y (float): The y-coordinate of the target point.
        - desired_speed (float): The desired speed of the agent.

        Returns:
        - None
        """
        # Calculate the error in position
        error_x = target_x - self.x
        error_y = target_y - self.y

        # Calculate the distance to the target
        dist_to_target = np.sqrt(error_x**2 + error_y**2)

        # Calculate the desired velocity based on the error
        desired_velocity = self.kp * dist_to_target

        # Calculate the angle to the target
        target_angle = np.arctan2(error_y, error_x)

        # Calculate the error in angle
        error_angle = target_angle - self.theta

        # Ensure the error_angle is within the range [-pi, pi]
        if error_angle > np.pi:
            error_angle -= 2 * np.pi
        elif error_angle < -np.pi:
            error_angle += 2 * np.pi

        # Calculate the desired angular velocity based on the angle error
        desired_angular_velocity = self.kp * error_angle

        # Set the desired linear and angular velocities
        self.v = desired_velocity
        self.omega = desired_angular_velocity

        # Limit the linear velocity to the desired speed
        if self.v > desired_speed:
            self.v = desired_speed

    def get_sensor_data(self, sensor_name):
        """
        Returns the data from a sensor.

        Args:
        - sensor_name (str): The name of the sensor.

        Returns:
        - float: The data from the sensor.
        """
        return self.sensors[sensor_name].get_data()
        