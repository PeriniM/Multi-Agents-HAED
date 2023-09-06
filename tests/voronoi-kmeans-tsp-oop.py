import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from python_tsp.heuristics import solve_tsp_simulated_annealing
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


class RoomMap:
    def __init__(self, filename):
        self.filename = filename
        self.shapes = []
        self.room_shape = []
        self.import_csv()

    def import_csv(self):
        df = pd.read_csv(self.filename, header=None, skiprows=1)
        shapes_coord = []
        for i in range(len(df)):
            x = np.fromstring(df.iloc[i, 1], sep=' ')
            y = np.fromstring(df.iloc[i, 2], sep=' ') * -1
            element_type = df.iloc[i, 4]
            shapes_coord.append([x, y, element_type])

            if element_type == 'room':
                x_max = np.max(x)
                y_max = np.max(y)
                x_min = np.min(x)
                y_min = np.min(y)

        width_room = abs(x_max - x_min)
        height_room = abs(y_max - y_min)

        x_scale = [0, width_room]
        y_scale = [0, height_room]

        for shape in shapes_coord:
            shape[0] = (shape[0] - x_min) * (x_scale[1] - x_scale[0]) / (x_max - x_min) + x_scale[0]
            shape[1] = (shape[1] - y_min) * (y_scale[1] - y_scale[0]) / (y_max - y_min) + y_scale[0]

        self.shapes = shapes_coord
        self.room_shape = [shape for shape in shapes_coord if shape[2] == "room"][0]

    def display(self):
        fig, ax = plt.subplots()
        for shape in self.shapes:
            x = np.append(shape[0], shape[0][0])
            y = np.append(shape[1], shape[1][0])
            element_type = shape[2]

            if element_type == 'room':
                ax.plot(x, y, color="blue", label="Room")
            elif element_type == 'agent':
                ax.plot(x, y, color="red", label="Agent")
            elif element_type == 'obstacle':
                ax.plot(x, y, color="green", label="Obstacle")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Room with Obstacles and Agents')
        ax.legend(loc="upper right")
        plt.show()


class VoronoiHandler:
    def __init__(self, room_shape):
        self.room_shape = room_shape
        self.vor = None
        self.clipped_ridges = []

    def generate_voronoi_within_room(self, n_points=1000):
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
        self.vor = Voronoi(points)
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
        fig, ax = plt.subplots()
        for line in self.clipped_ridges:
            x, y = line.xy
            ax.plot(x, y, color="black")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Clipped Voronoi Ridges')
        plt.show()

class RobotAssigner:
    def __init__(self, vor, room_shape, n_robots):
        self.vor = vor
        self.room_shape = room_shape
        self.n_robots = n_robots
        self.robot_assignments = []

    def divide_areas_using_kmeans(self):
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
        print(f"Dividing {len(region_polygons)} regions into {self.n_robots} robots...")
        kmeans = KMeans(n_clusters=self.n_robots).fit(centroids)
        labels = kmeans.labels_

        self.robot_assignments = [[] for _ in range(self.n_robots)]
        for i, region in enumerate(region_polygons):
            self.robot_assignments[labels[i]].append(region)

    @staticmethod
    def are_neighbors(region1, region2):
        shared_boundary = region1.boundary.intersection(region2.boundary)
        return shared_boundary.length > 0

    def compute_tsp_path_for_region(self, regions):
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
        print(f"Computing TSP path for {len(regions)} regions...")
        tour, _ = solve_tsp_simulated_annealing(distance_matrix)
        return tour

    def plot_assignments_and_paths(self):
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


if __name__ == "__main__":
    # Initializing the RoomMap
    room_map = RoomMap('Rooms/createGrid().csv')
    room_map.display()

    # Handling Voronoi tessellation
    voronoi_handler = VoronoiHandler(room_map.room_shape)
    voronoi_handler.generate_voronoi_within_room(300)
    voronoi_handler.plot_clipped_ridges()

    # Assigning regions to robots and plotting
    robot_assigner = RobotAssigner(voronoi_handler.vor, room_map.room_shape, n_robots=4)
    robot_assigner.divide_areas_using_kmeans()
    robot_assigner.plot_assignments_and_paths()
