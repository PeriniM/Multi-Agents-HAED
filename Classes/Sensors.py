class Sensor:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # def addUncertainty(self, uncertainty):
    #     # Implement uncertainty logic here


class UWBAnchor(Sensor):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.robot_distance = None

    def getDistance(self, robot_coords):
        dx = self.x - robot_coords[0]
        dy = self.y - robot_coords[1]
        self.robot_distance = (dx**2 + dy**2)**0.5

        return self.robot_distance

# class Magnetometer(Sensor):
#     def __init__(self):
#         super().__init__()

# class Lidar(Sensor):
#     def __init__(self):
#         super().__init__()

# class StereoCamera(Sensor):
#     def __init__(self):
#         super().__init__()