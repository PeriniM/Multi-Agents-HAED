import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from python_tsp.heuristics import solve_tsp_simulated_annealing
import warnings
import time

# Suppress FutureWarning and RuntimeWarning
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

class RobotAssigner:
    """
    A class for assigning robots to regions using Voronoi diagrams and solving the Traveling Salesman Problem (TSP)
    for each robot's assigned regions.

    Attributes:
        vor (scipy.spatial.Voronoi): Voronoi diagram of the room.
        room_shape (tuple): Tuple of two arrays representing the x and y coordinates of the room's shape.
        n_robots (int): Number of robots to assign.
        robot_assignments (list): List of lists representing the regions assigned to each robot.

    Methods:
        divide_areas_using_kmeans(): Divides the Voronoi cells among the robots using KMeans clustering.
        are_neighbors(region1, region2): Returns True if region1 and region2 share a boundary, False otherwise.
        compute_tsp_path_for_region(regions): Computes the TSP path for a list of regions.
        compute_tsp_paths(): Computes the TSP paths for all the regions assigned to each robot.
        plot_assignments_and_paths(): Plots the robot assignments and TSP paths.
    """

    def __init__(self, vor, room_shape, n_robots):
        """
        Initializes a RobotAssigner object.

        Args:
            vor (scipy.spatial.Voronoi): Voronoi diagram of the room.
            room_shape (tuple): Tuple of two arrays representing the x and y coordinates of the room's shape.
            n_robots (int): Number of robots to assign.
        """
        self.vor = vor
        self.room_shape = room_shape
        self.n_robots = n_robots
        self.robot_assignments = []

    def divide_areas_using_kmeans(self):
        """
        Divides the Voronoi cells among the robots using KMeans clustering.
        """
        centroids = []
        region_polygons = []
        room_polygon = Polygon(zip(self.room_shape[0], self.room_shape[1]))

        for point_region in self.vor.point_region:
            region = self.vor.regions[point_region]
            if -1 not in region:
                polygon_vertices = [self.vor.vertices[i] for i in region]
                region_polygon = Polygon(polygon_vertices).intersection(room_polygon)
                if not region_polygon.is_empty:
                    centroids.append(np.array(region_polygon.centroid.coords).squeeze())
                    region_polygons.append(region_polygon)
        print(f"Dividing {len(region_polygons)} voronoi cells among {self.n_robots} robots...")
        prev_time = time.time()
        kmeans = KMeans(n_clusters=self.n_robots).fit(centroids)
        print(f"{round((time.time() - prev_time)*1000, 3)} ms")
        labels = kmeans.labels_

        self.robot_assignments = [[] for _ in range(self.n_robots)]
        for i, region in enumerate(region_polygons):
            self.robot_assignments[labels[i]].append(region)

    @staticmethod
    def are_neighbors(region1, region2):
        """
        Returns True if region1 and region2 share a boundary, False otherwise.

        Args:
            region1 (shapely.geometry.Polygon): First region.
            region2 (shapely.geometry.Polygon): Second region.

        Returns:
            bool: True if region1 and region2 share a boundary, False otherwise.
        """
        shared_boundary = region1.boundary.intersection(region2.boundary)
        return shared_boundary.length > 0

    def compute_tsp_path_for_region(self, regions):
        """
        Computes the TSP path for a list of regions.

        Args:
            regions (list): List of shapely.geometry.Polygon objects representing the regions.

        Returns:
            list: List of indices representing the order in which the regions should be visited.
        """
        centroids = [region.centroid.coords[0] for region in regions]
        num_points = len(centroids)
        distance_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                distance = Point(centroids[i]).distance(Point(centroids[j]))
                if i != j:
                    if self.are_neighbors(regions[i], regions[j]):
                        distance *= 0.5
                    else:
                        distance *= 1e5
                distance_matrix[i, j] = distance
        print(f"Computing TSP path for {len(regions)} voronoi cells...")
        prev_time = time.time()
        tour, _ = solve_tsp_simulated_annealing(distance_matrix)
        print(f"{round((time.time() - prev_time)*1000, 3)} ms")
        return tour
    
    def compute_tsp_paths(self):
        """
        Computes the TSP paths for all the regions assigned to each robot.

        Returns:
            list: List of lists representing the TSP paths for each robot.
        """
        region_tours = []
        for robot_idx, regions in enumerate(self.robot_assignments):
            tour = self.compute_tsp_path_for_region(regions)
            ordered_regions = [regions[i] for i in tour]
            x_vals, y_vals = [], []
            for region in ordered_regions:
                x, y = region.centroid.coords[0]
                x_vals.append(x)
                y_vals.append(y)
            region_tours.append([x_vals, y_vals])
        return region_tours

    def plot_assignments_and_paths(self):
        """
        Plots the robot assignments and TSP paths.
        """
        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, self.n_robots))
        for robot_idx, regions in enumerate(self.robot_assignments):
            tour = self.compute_tsp_path_for_region(regions)
            ordered_regions = [regions[i] for i in tour]
            x_vals, y_vals = [], []
            for region in ordered_regions:
                x, y = region.centroid.coords[0]
                x_vals.append(x)
                y_vals.append(y)
            ax.plot(x_vals, y_vals, marker='o', color=colors[robot_idx], label=f"Robot {robot_idx}")

        ax.set_title('Robot Assignments with TSP Paths')
        ax.legend()
        plt.show()