import numpy as np
from Classes2.Shapes import Shape

class Agent(Shape):
    def __init__(self, x, y):
        super().__init__(x, y, element_type="agent")
        self.x_ideal = x
        self.y_ideal = y
        self.CoM_ideal = np.array([np.mean(x), np.mean(y)])
        self.x_estim = None
        self.y_estim = None
        self.CoM_estim = None
        self.orientation_estim = None
        self.dynamics = "unicycle"
        self.theta = np.random.rand() * 2 * np.pi
        self.max_speed = 100  # meters per second
        self.max_angular_velocity = np.pi * 10  # radians per second
    
    def updateDynamics(self, dt):
        # Generate random velocity and angular velocity for the agent
        speed = np.random.rand() * self.max_speed
        angular_velocity = (np.random.rand() - 0.5) * 2 * self.max_angular_velocity

        # Calculate the x and y components of the velocity and orientation
        vx = speed * np.cos(self.theta)
        vy = speed * np.sin(self.theta)
        omega = angular_velocity

        # Update the position and orientation of the agent
        self.CoM_estim[0] += vx * dt
        self.CoM_estim[1] += vy * dt
        self.theta += omega * dt

    def updatePosition(self, dt, velocity, angular_velocity):
        # Update the position and orientation of the agent based on provided velocity and angular velocity
        vx = velocity * np.cos(self.theta)
        vy = velocity * np.sin(self.theta)
        omega = angular_velocity

        self.CoM_estim[0] += vx * dt
        self.CoM_estim[1] += vy * dt
        self.theta += omega * dt
        