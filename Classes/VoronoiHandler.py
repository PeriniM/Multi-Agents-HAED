import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import cascaded_union
import time

class VoronoiHandler:
    """
    A class to handle Voronoi diagrams within a given room shape.

    ...

    Attributes
    ----------
    room_shape : tuple
        A tuple containing the x and y coordinates of the room shape.
    vor : Voronoi object
        A Voronoi object representing the Voronoi diagram generated within the room shape.
    clipped_ridges : list
        A list of LineString objects representing the clipped Voronoi ridges within the room shape.

    Methods
    -------
    generate_voronoi_within_room(n_points=1000)
        Generates a Voronoi diagram within the room shape using a specified number of points.
    plot_clipped_ridges()
        Plots the clipped Voronoi ridges within the room shape.
    """
    def __init__(self, room_shape):
        """
        Parameters
        ----------
        room_shape : tuple
            A tuple containing the x and y coordinates of the room shape.
        """
        self.room_shape = room_shape
        self.vor = None
        self.clipped_ridges = []

    def generate_voronoi_within_room(self, n_points=1000):
        """
        Generates a Voronoi diagram within the room shape using a specified number of points.

        Parameters
        ----------
        n_points : int, optional
            The number of points to use for generating the Voronoi diagram. Default is 1000.
        """
        room_x = self.room_shape[0]
        room_y = self.room_shape[1]
        points = []
        while len(points) < n_points:
            x = np.random.uniform(min(room_x), max(room_x))
            y = np.random.uniform(min(room_y), max(room_y))
            path = patches.Path(np.vstack((room_x, room_y)).T)
            if path.contains_point((x, y)):
                points.append([x, y])
        print(f"Generating Voronoi with {len(points)} points...")
        points = np.array(points)
        prev_time = time.time()
        self.vor = Voronoi(points)
        print(f"{round((time.time() - prev_time)*1000, 3)} ms")
        room_polygon = Polygon(zip(self.room_shape[0], self.room_shape[1]))

        for ridge_vertices in self.vor.ridge_vertices:
            if -1 not in ridge_vertices:
                line = LineString([self.vor.vertices[i] for i in ridge_vertices])
                intersection = room_polygon.intersection(line)
                if intersection.geom_type == "LineString":
                    self.clipped_ridges.append(intersection)
                elif intersection.geom_type == "MultiLineString":
                    self.clipped_ridges.extend(list(intersection.geoms))
            
    def plot_clipped_ridges(self):
        """
        Plots the clipped Voronoi ridges within the room shape.
        """
        fig, ax = plt.subplots()
        for line in self.clipped_ridges:
            x, y = line.xy
            ax.plot(x, y, color="black")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Clipped Voronoi Ridges')
        plt.show()