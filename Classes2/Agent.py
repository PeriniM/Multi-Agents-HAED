import numpy as np
from Classes2.Shapes import Shape

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
        self.max_v = 5
        self.max_omega = 5

        # set wheels distance as the minimum distance between the middle point and the closest vertex
        self.wheels_distance = np.min(np.sqrt((self.x - vertex_x)**2 + (self.y - vertex_y)**2))
        self.wheel_radius = self.wheels_distance / 2

        self.dynamics = "differential"
        self.sensors = {}

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
    
    def move(self, left_speed, right_speed, dt):
        # Update the agent's pose
        if self.dynamics == "differential":
            # Calculate the new pose
            delta_s = 0.5 * (left_speed + right_speed) * dt
            delta_theta = (right_speed - left_speed) / self.wheels_distance * dt
            self.x += delta_s * np.cos(self.theta + 0.5 * delta_theta)
            self.y += delta_s * np.sin(self.theta + 0.5 * delta_theta)
            self.theta += delta_theta
        else:
            raise ValueError("Unknown dynamics type: {}".format(self.dynamics))

    def get_sensor_data(self, sensor_name):
        return self.sensors[sensor_name].get_data()
        