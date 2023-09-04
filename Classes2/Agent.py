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

        # set wheels distance as the minimum distance between the middle point and the closest vertex
        self.wheels_distance = np.min(np.sqrt((self.x - vertex_x)**2 + (self.y - vertex_y)**2))
        self.wheel_radius = self.wheels_distance / 2

        self.dynamics = "differential"
        self.sensors = {}

    def getVertices(self):
        # return transformed vertices
        new_vertex_x = self.x + (self.vertex_x - self.x) * np.cos(self.theta) + (self.vertex_y - self.y) * np.sin(self.theta)
        new_vertex_y = self.y - (self.vertex_x - self.x) * np.sin(self.theta) + (self.vertex_y - self.y) * np.cos(self.theta)
        return np.array([new_vertex_x, new_vertex_y]).T
    
    def initialize_sensors(self, sensors):
        for sensor in sensors:
            key = type(sensor).__name__
            # If it's an encoder, specify which wheel
            if key == "Encoder":
                key += "_left" if "Encoder_left" not in self.sensors else "_right"
            self.sensors[key] = sensor
    
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
        
        # Update all sensors
        for sensor in self.sensors.values():
            if sensor.sensor_type == "EncoderLeft":
                sensor.update(self.wheel_radius, left_speed, dt)
            elif sensor.sensor_type == "EncoderRight":
                sensor.update(self.wheel_radius, right_speed, dt)
            elif sensor.sensor_type == "Gyroscope":
                # Update with the angular velocity, which is delta_theta/dt
                sensor.update((right_speed - left_speed) / self.wheels_distance)
            elif sensor.sensor_type == "Accelerometer":
                # Update with the linear acceleration
                # This is a simplified example; consider using a more accurate model.
                delta_v = 0.5 * (right_speed - left_speed)
                sensor.update([delta_v/dt, 0])
            elif sensor.sensor_type == "Magnetometer":
                # Update with a "true" magnetic field value. This is an example.
                # In a more complex scenario, you might simulate Earth's magnetic field and other magnetic disturbances.
                sensor.update([np.cos(self.theta), np.sin(self.theta)])
            elif sensor.sensor_type == "UWBAnchor":
                sensor.update(self.x, self.y)
    
    def get_sensor_data(self, sensor_name):
        # For encoders, you can retrieve data as:
        # agent.get_sensor_data("Encoder_left")
        # agent.get_sensor_data("Encoder_right")
        return self.sensors[sensor_name].get_data()
        