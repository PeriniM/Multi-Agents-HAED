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

    def getVertices(self):
        return np.array([self.x, self.y]).T 

class Room(Shape):
    def __init__(self, x, y):
        super().__init__(x, y, element_type='room', closed=True)        
        self.obstacles = []
    
    def addObstacle(self, obstacle):
        self.obstacles.append(obstacle)

class Obstacle(Shape):
    def __init__(self, x, y):
        super().__init__(x, y, element_type='obstacle', closed=True)