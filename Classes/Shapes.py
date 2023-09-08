import numpy as np

class Shape():
    def __init__(self, vertex_x, vertex_y, element_type, closed=False):
        """
        Initializes a Shape object with the given vertices and element type.

        Args:
        vertex_x (numpy.ndarray): An array of x-coordinates of the vertices.
        vertex_y (numpy.ndarray): An array of y-coordinates of the vertices.
        element_type (str): The type of the shape element.
        closed (bool, optional): Whether the shape is closed or not. Defaults to False.
        """
        self.vertex_x = vertex_x
        self.vertex_y = vertex_y
        self.element_type = element_type
        self.closed = closed
        # if closed, then append the first point to the end of the array
        if closed:
            self.vertex_x = np.append(self.vertex_x, self.vertex_x[0])
            self.vertex_y = np.append(self.vertex_y, self.vertex_y[0])      

    def getVertices(self):
        """
        Returns an array of vertices of the shape.

        Returns:
        numpy.ndarray: An array of vertices of the shape.
        """
        return np.array([self.vertex_x, self.vertex_y]).T 
    
    def getLines(self):
        """
        Returns an array of line segments of the shape.

        Returns:
        list: An array of line segments of the shape.
        """
        segments = []
        for i in range(len(self.vertex_x) - 1):  # Subtracting 1 to avoid an index out of range error
            segment = (self.vertex_x[i], self.vertex_y[i], self.vertex_x[i+1], self.vertex_y[i+1])
            segments.append(segment)
        return segments

class Room(Shape):
    """
    A class representing a room shape, which is a closed shape with obstacles inside.
    
    Attributes:
    - vertex_x (int): the x-coordinate of the vertex of the room
    - vertex_y (int): the y-coordinate of the vertex of the room
    - element_type (str): the type of the shape, which is 'room' for a room shape
    - closed (bool): a flag indicating whether the shape is closed or not
    - obstacles (list): a list of obstacles inside the room shape
    """
    def __init__(self, vertex_x, vertex_y):
        super().__init__(vertex_x, vertex_y, element_type='room', closed=True)        
        self.obstacles = []
    
    def addObstacle(self, obstacle):
        self.obstacles.append(obstacle)

class Obstacle(Shape):
    """
    A class representing an obstacle shape.

    Attributes:
    - vertex_x (float): The x-coordinate of the vertex of the obstacle.
    - vertex_y (float): The y-coordinate of the vertex of the obstacle.
    - element_type (str): The type of the shape element. Default is 'obstacle'.
    - closed (bool): Whether the shape is closed or not. Default is True.
    """
    def __init__(self, vertex_x, vertex_y):
        super().__init__(vertex_x, vertex_y, element_type='obstacle', closed=True)