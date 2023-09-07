import numpy as np
from Classes.Shapes import Shape
from Classes.AgentEKF import AgentEKF

class Agent(Shape):
    def __init__(self, vertex_x, vertex_y):
        super().__init__(vertex_x, vertex_y, element_type="agent")
        self.vertex_x = vertex_x
        self.vertex_y = vertex_y
        self.x = np.mean(vertex_x)
        self.y = np.mean(vertex_y)
        self.theta = 0
        self.v = 0
        self.omega = 0

        # set max speed and angular velocity
        self.max_v = 2
        self.max_omega = 2
        # set PID constants
        self.kp = 1000

        # set wheels distance as the minimum distance between the middle point and the closest vertex
        self.wheels_distance = np.min(np.sqrt((self.x - vertex_x)**2 + (self.y - vertex_y)**2))
        self.wheel_radius = self.wheels_distance / 2

        self.dynamics = "differential"
        self.sensors = {}

        # Initialize the Extended Kalman Filter for the agent
        self.ekf = None

        self.estim_pos_uwb = None
        self.estim_pos_encoders = None

        self.estim_v_uwb = None
        self.estim_v_encoders = None

        self.estim_omega_uwb = None
        self.estim_omega_encoders = None

        self.estim_theta_uwb = None
        self.estim_theta_encoders = None

        # Iniitialize the target points of the path to follow 
        self.target_points = None
        self.scanned_map = []
        
    def getVertices(self):
        # return transformed vertices
        new_vertex_x = self.x + (self.vertex_x - self.x) * np.cos(self.theta) + (self.vertex_y - self.y) * np.sin(self.theta)
        new_vertex_y = self.y - (self.vertex_x - self.x) * np.sin(self.theta) + (self.vertex_y - self.y) * np.cos(self.theta)
        return np.array([new_vertex_x, new_vertex_y]).T
    
    def initialize_sensors(self, sensors):
        for sensor in sensors:
            if sensor.sensor_type == "Encoder":
                self.sensors["Encoder_left"] = sensor
                self.sensors["Encoder_right"] = sensor
            else:
                self.sensors[sensor.sensor_type] = sensor
        
    def initialize_ekf(self, encoder_noise, uwb_noise):
        self.ekf = AgentEKF(encoder_noise, uwb_noise)
        # Setting the starting estimate for theta in the EKF to match the agent's orientation
        self.ekf.x[2] = self.theta

    def HJacobian_at(self, x):
        return np.eye(4)  # Jacobian matrix for the linear measurement function

    def Hx_at(self, x):
        return x  # Direct observation of state
    
    def move(self, target_x, target_y, desired_speed, dt):
        # Use the proportional controller to update velocities
        self.update(target_x, target_y, desired_speed, dt)

        # Ensure linear and angular velocities are within limits
        self.v = np.clip(self.v, -self.max_v, self.max_v)
        self.omega = np.clip(self.omega, -self.max_omega, self.max_omega)
        
        # Update the agent's pose
        if self.dynamics == "differential":
            # Calculate the new pose
            delta_s = self.v * dt
            delta_theta = self.omega * dt
            self.x += delta_s * np.cos(self.theta + 0.5 * delta_theta)
            self.y += delta_s * np.sin(self.theta + 0.5 * delta_theta)
            self.theta += delta_theta
        else:
            raise ValueError("Unknown dynamics type: {}".format(self.dynamics))
    
    def update(self, target_x, target_y, desired_speed, dt):
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
        return self.sensors[sensor_name].get_data()
        