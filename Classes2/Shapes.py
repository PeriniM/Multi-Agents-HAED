import numpy as np

class Shape():
    def __init__(self, vertex_x, vertex_y, element_type, closed=False):
        self.vertex_x = vertex_x
        self.vertex_y = vertex_y
        self.element_type = element_type
        self.closed = closed
        # if closed, then append the first point to the end of the array
        if closed:
            self.vertex_x = np.append(self.vertex_x, self.vertex_x[0])
            self.vertex_y = np.append(self.vertex_y, self.vertex_y[0])      

    def getVertices(self):
        return np.array([self.vertex_x, self.vertex_y]).T 
    
    def getLines(self):
        segments = []
        for i in range(len(self.vertex_x) - 1):  # Subtracting 1 to avoid an index out of range error
            segment = (self.vertex_x[i], self.vertex_y[i], self.vertex_x[i+1], self.vertex_y[i+1])
            segments.append(segment)
        return segments

class Room(Shape):
    def __init__(self, vertex_x, vertex_y):
        super().__init__(vertex_x, vertex_y, element_type='room', closed=True)        
        self.obstacles = []
    
    def addObstacle(self, obstacle):
        self.obstacles.append(obstacle)

class Obstacle(Shape):
    def __init__(self, vertex_x, vertex_y):
        super().__init__(vertex_x, vertex_y, element_type='obstacle', closed=True)