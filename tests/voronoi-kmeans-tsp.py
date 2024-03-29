import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from itertools import product

from sklearn.cluster import KMeans
from python_tsp.heuristics import solve_tsp_simulated_annealing
import warnings
import time

# Suppress FutureWarning and RuntimeWarning
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

def importCSV(filename):
    df = pd.read_csv(filename, header=None, skiprows=1)
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

    # normalize the vertices of the room, obstacles and agents
    for shape in shapes_coord:
        shape[0] = (shape[0] - x_min) * (x_scale[1] - x_scale[0]) / (x_max - x_min) + x_scale[0]
        shape[1] = (shape[1] - y_min) * (y_scale[1] - y_scale[0]) / (y_max - y_min) + y_scale[0]

    return shapes_coord

def display_shapes(shapes):
    fig, ax = plt.subplots()
    
    for shape in shapes:
        x = np.append(shape[0], shape[0][0])  # Connect the last and first points
        y = np.append(shape[1], shape[1][0])  # Connect the last and first points
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

def generate_voronoi_within_room(room_shape, n_points=1000):
    room_x = room_shape[0]
    room_y = room_shape[1]
    
    # Generate random points inside room
    points = []
    while len(points) < n_points:
        x = np.random.uniform(min(room_x), max(room_x))
        y = np.random.uniform(min(room_y), max(room_y))
        
        # Check if the point is inside the room
        path = patches.Path(np.vstack((room_x, room_y)).T)
        if path.contains_point((x, y)):
            points.append([x, y])
    
    points = np.array(points)
    print(f"Generating Voronoi with {len(points)} points...")
    prev_time = time.time()
    vor = Voronoi(points)
    print(f"{round((time.time() - prev_time)*1000, 3)} ms")
    # Convert the room shape to a Shapely Polygon
    room_polygon = Polygon(zip(room_shape[0], room_shape[1]))

    # Identify and store Voronoi ridges that are within the room
    clipped_ridges = []
    for ridge_vertices in vor.ridge_vertices:
        if -1 not in ridge_vertices:  # Exclude ridges that go to infinity
            line = LineString([vor.vertices[i] for i in ridge_vertices])
            intersection = room_polygon.intersection(line)
            
            if intersection.geom_type == "LineString":
                clipped_ridges.append(intersection)
            elif intersection.geom_type == "MultiLineString":
                clipped_ridges.extend(list(intersection.geoms))
    
    return vor, clipped_ridges

def divide_areas_using_kmeans(vor, room_shape, n_robots):
    centroids = []
    region_polygons = []

    room_polygon = Polygon(zip(room_shape[0], room_shape[1]))

    # Compute the region polygons and their centroids
    for point_region in vor.point_region:
        region = vor.regions[point_region]
        if -1 not in region:  # Ensure it doesn't go to infinity
            polygon_vertices = [vor.vertices[i] for i in region]
            region_polygon = Polygon(polygon_vertices).intersection(room_polygon)
            
            if not region_polygon.is_empty:
                centroids.append(np.array(region_polygon.centroid.coords).squeeze())
                region_polygons.append(region_polygon)
                
    # Cluster the centroids using K-means
    print(f"Dividing {len(centroids)} voronoi cells among {n_robots} robots...")
    prev_time = time.time()
    kmeans = KMeans(n_clusters=n_robots).fit(centroids)
    print(f"{round((time.time() - prev_time)*1000, 3)} ms")
    labels = kmeans.labels_

    robot_assignments = [[] for _ in range(n_robots)]
    for i, region in enumerate(region_polygons):
        robot_assignments[labels[i]].append(region)

    return robot_assignments

def are_neighbors(region1, region2):
    shared_boundary = region1.boundary.intersection(region2.boundary)
    return shared_boundary.length > 0

def compute_tsp_path_for_region(regions):
    centroids = [region.centroid.coords[0] for region in regions]

    num_points = len(centroids)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            distance = Point(centroids[i]).distance(Point(centroids[j]))
            if i != j:
                if are_neighbors(regions[i], regions[j]):
                    distance *= 0.5  # Decrease the distance for neighboring cells to prioritize them
                else:
                    distance *= 1e5  # Increase the distance for non-neighboring cells to deprioritize them
            distance_matrix[i][j] = distance

    # Solve the TSP problem
    print(f"Computing TSP path for {num_points} voronoi cells...")
    prev_time = time.time()
    permutation, _ = solve_tsp_simulated_annealing(distance_matrix)
    print(f"{round((time.time() - prev_time)*1000, 3)} ms")
    # Get the ordered centroid coordinates based on the TSP path
    ordered_centroids = [centroids[i] for i in permutation]
    
    return ordered_centroids


if __name__ == "__main__":
    filename = 'Rooms/noObstacles.csv'
    shapes = importCSV(filename)
    
    # Extract the room shape
    room_shape = [shape for shape in shapes if shape[2] == "room"][0]

    # Get the Voronoi object and the clipped ridges
    vor, clipped_ridges = generate_voronoi_within_room(room_shape, 300)

    # Use K-means to split the Voronoi regions among robots
    n_robots = 4  # change this as per your needs
    robot_assignments1 = divide_areas_using_kmeans(vor, room_shape, 1)
    robot_assignments2 = divide_areas_using_kmeans(vor, room_shape, 5)
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    
    # # Plot the Voronoi cells with different colors for each robot assignment
    colors1 = cm.rainbow(np.linspace(0, 1, len(robot_assignments1)))
    colors2 = cm.rainbow(np.linspace(0, 1, len(robot_assignments2)))
    # for i, regions in enumerate(robot_assignments):
    #     for region in regions:
    #         # Check if the region is a MultiPolygon
    #         if region.geom_type == 'MultiPolygon':
    #             for sub_region in region.geoms:
    #                 x, y = sub_region.exterior.xy
    #                 ax.fill(x, y, color=colors[i], alpha=0.5)
    #         else:
    #             x, y = region.exterior.xy
    #             ax.fill(x, y, color=colors[i], alpha=0.5)
    
    # Compute TSP paths for each robot assignment and plot
    for i, regions in enumerate(robot_assignments1):
        ordered_centroids = compute_tsp_path_for_region(regions)
        ax[0].plot(*zip(*ordered_centroids), '-o', color=colors1[i], linewidth=2, markersize=5, label=f"Robot {i+1}")
    # Compute TSP paths for each robot assignment and plot
    for i, regions in enumerate(robot_assignments2):
        ordered_centroids = compute_tsp_path_for_region(regions)
        ax[1].plot(*zip(*ordered_centroids), '-o', color=colors2[i], linewidth=2, markersize=5, label=f"Robot {i+1}")
    # # Plot the clipped Voronoi ridges
    # for ridge in clipped_ridges:
    #     if ridge.geom_type == 'LineString':
    #         ax.plot(*ridge.xy, color='k', linewidth=0.5)
    #     elif ridge.geom_type == 'MultiLineString':
    #         for line in ridge:
    #             ax.plot(*line.xy, color='k', linewidth=0.5)
    
    # Plot room boundary
    room_x = np.append(room_shape[0], room_shape[0][0])
    room_y = np.append(room_shape[1], room_shape[1][0])
    ax[0].plot(room_x, room_y, color="k")
    ax[1].plot(room_x, room_y, color="k")
    
    # Set the plot limits to show the entire room
    ax[1].set_xlim(min(room_x) - 1, max(room_x) + 1)  # Adding/subtracting 1 for a little margin
    ax[1].set_ylim(min(room_y) - 1, max(room_y) + 1)
    
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[0].set_title('TSP Paths Comparison')
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    
    plt.show()
