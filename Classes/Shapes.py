import numpy as np

class Shape:
    def __init__(self, x, y, element_type, closed=False):
        self.x = x
        self.y = y
        self.element_type = element_type
        self.closed = closed
        # if closed, then append the first point to the end of the array
        if closed:
            self.x = np.append(self.x, self.x[0])
            self.y = np.append(self.y, self.y[0])        

class Room(Shape):
    def __init__(self, x, y):
        super().__init__(x, y, element_type='room', closed=True)        
        
class Obstacle(Shape):
    def __init__(self, x, y):
        super().__init__(x, y, element_type='obstacle', closed=True)

class Agent(Shape):
    def __init__(self, x, y):
        super().__init__(x, y, element_type="agent")
        self.x_real = x
        self.y_real = y
        self.x_estim = None
        self.y_estim = None
        self.CoM_estim = None
        self.orientation_estim = None
        self.dynamics = None
    
    def getRealCoM(self):
        return np.mean(np.array([self.x_real, self.y_real]), axis=1)
    
#     def addDynamics(self, max_speed, max_angular_velocity, dt):
#         # Implement dynamics logic here
        
#     def getMagnetometer(self, north_pole):
#         # Implement magnetometer logic here