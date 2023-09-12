import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

def compute_histograms(polygon, reverse=False):
    def angle_between_lines(p1, p2, p3):
        """p1: (x1, y1), p2: (x2, y2), p3: (x3, y3)
        Returns the angle between the lines p1p2 and p2p3 in degrees"""
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))
        if angle < 0:  # Convert negative angles to their positive equivalent
            angle += 360
        return angle

    n = len(polygon)
    angles = []
    side_lengths = []
    normals = []

    for i in range(n):
        if i == 0:
            prev_point = polygon[-1]
        else:
            prev_point = polygon[i-1]
        if i == n-1:
            next_point = polygon[0]
        else:
            next_point = polygon[i+1]

        # Get three consecutive points
        p1, p2, p3 = prev_point, polygon[i], next_point

        # Calculate the internal angle at p2
        angle = angle_between_lines(p1, p2, p3)

        angles.append(angle)

        # Calculate side length
        side = np.array(p3) - np.array(p2)
        length = np.linalg.norm(side)
        side_lengths.append(length)

        # Calculate normal orientation (assuming it points to the left of the direction of travel)
        normal = np.array([-side[1], side[0]])
        normal_orientation = np.degrees(np.arctan2(normal[1], normal[0]))
        if normal_orientation < 0:  # Convert negative angles to their positive equivalent
            normal_orientation += 360
        normals.append(normal_orientation)
    
    if reverse:
        angles = [360 - angle for angle in angles]

    return angles, side_lengths, normals

def plot_histograms(polygon, ax0, ax1, ax2, reverse=False, highlight_side=None):
    angles, side_lengths, normals = compute_histograms(polygon, reverse=reverse)
    
    # Plot the polygon itself
    x, y = zip(*polygon + [polygon[0]])  # Adding the first point to close the polygon
    for i, (x1, y1, x2, y2) in enumerate(zip(x[:-1], y[:-1], x[1:], y[1:])):
        if i == highlight_side:
            ax0.plot([x1, x2], [y1, y2], '-o', linewidth=2.5, markersize=7, color='red')
        else:
            ax0.plot([x1, x2], [y1, y2], '-o', linewidth=1.5, markersize=5, color='C0')
        # add side numbers to each side near the middle
        ax0.text((x1 + x2)/2, (y1 + y2)/2, str(i+1), ha="center", va="center", fontsize=8)
    ax0.set_title("Polygon")
    ax0.set_aspect('equal', 'box')

    side_lengths.append(side_lengths[-1])
    angles.append(angles[-1])
    normals.append(normals[-1])

    if reverse:
        angles = 360 - np.array(angles)
        normals = 360 - np.array(normals)

    edges = np.cumsum(side_lengths)  # Add a zero to the end to make the lengths and angles the same length
    # Angle Histogram
    ax1.fill_between(edges, angles, step="post", edgecolor='C1', facecolor='none', linewidth=1.5)
    # Add numbers to the top of each bar in the middle of the bin
    for i, (x, y) in enumerate(zip(edges, angles)):
        if i < len(edges) - 1:
            # small font size to fit the numbers in the bins
            ax1.text(x + side_lengths[i]/2, y, str(i+1), ha="center", va="bottom", fontsize=6)
            # show small certical lines to indicate the bin edges
            ax1.plot([x, x], [0, y], color='C1', linewidth=0.2)

    
    # add vertical lines to indicate the bin edges
    ax1.set_xticks(edges)
    ax1.set_xticklabels([str(i+1) for i in range(len(edges))])
    ax1.set_title("Angle Histogram")
    ax1.set_xlabel("Cumulative Length")
    ax1.set_ylabel("Angle (degrees)")

    # Normal Orientation Histogram
    ax2.fill_between(edges, normals, step="post", edgecolor='C2', facecolor='none', linewidth=1.5)
    # Add numbers to the top of each bar in the middle of the bin
    for i, (x, y) in enumerate(zip(edges, normals)):
        if i < len(edges) - 1:
            # small font size to fit the numbers in the bins
            ax2.text(x + side_lengths[i]/2, y, str(i+1), ha="center", va="bottom", fontsize=6)
            # show small certical lines to indicate the bin edges
            ax2.plot([x, x], [0, y], color='C2', linewidth=0.2)
    
    # add vertical lines to indicate the bin edges
    ax2.set_xticks(edges)
    ax2.set_xticklabels([str(i+1) for i in range(len(edges))])
    ax2.set_title("Normal Orientation Histogram")
    ax2.set_xlabel("Cumulative Length")
    ax2.set_ylabel("Normal Orientation (degrees)")

def compute_all_differences(polygon1, polygon2):
    angles1, _, _ = compute_histograms(polygon1)
    angles2, _, _ = compute_histograms(polygon2, reverse=True)
    
    difference_matrix = []

    for angle2 in angles2:
        row = []
        for angle1 in angles1:
            diff = abs(angle2 - angle1)
            row.append(diff)
        difference_matrix.append(row)

    return np.array(difference_matrix)

def plot_3d_difference(polygon1, polygon2):
    difference_matrix = compute_all_differences(polygon1, polygon2)

    x = np.arange(1, len(polygon1) + 1)
    y = np.arange(1, len(polygon2) + 1)
    x, y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, difference_matrix, cmap='viridis', edgecolor='blue')
    
    ax.set_xlabel('Polygon1 Edges')
    ax.set_ylabel('Polygon2 Edges')
    ax.set_zlabel('Angle Difference (degrees)')
    ax.set_title("Angle Difference between Polygon Edges")
    # Adjusting x-axis tick labels to start from 1
    ax.set_xticks(np.arange(len(polygon1)))
    ax.set_xticklabels([str(i+1) for i in range(len(polygon1))])
    ax.set_yticks(np.arange(len(polygon2)))
    ax.set_yticklabels([str(i+1) for i in range(len(polygon2))])

    plt.show()

def plot_heatmap_difference(polygon1, polygon2):
    difference_matrix = compute_all_differences(polygon1, polygon2)
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.heatmap(difference_matrix, cmap='inferno', annot=True, fmt=".1f")
    
    ax.set_xlabel('Polygon1 Edges')
    ax.set_ylabel('Polygon2 Edges')
    ax.set_title("Angle Difference between Polygon Edges")
    
    # Adjusting x-axis tick labels to start from 1
    ax.set_xticks(np.arange(len(polygon1)))
    ax.set_xticklabels([str(i+1) for i in range(len(polygon1))])
    ax.set_yticks(np.arange(len(polygon2)))
    ax.set_yticklabels([str(i+1) for i in range(len(polygon2))])
    ax.invert_yaxis()  # To show the 1st edge of Polygon2 at the top
    
    plt.show()

polygon1 = [(1,3), (2,3), (2,2), (3,2), (3,1), (2,1), (2,0), (1,0), (1,1), (0,1), (0,2), (1,2)]
polygon2 = [(7,3), (8,3), (8,2), (9,2),(9,3), (10,3), (10,1), (7,1)]

fig, axs = plt.subplots(2, 3)

# Plot Histograms for each shape
plot_histograms(polygon1, axs[0, 0], axs[0, 1], axs[0, 2])
plot_histograms(polygon2, axs[1, 0], axs[1, 1], axs[1, 2] , reverse=False)

plt.tight_layout()
plt.show()

# Plot the difference between the two shapes angles
plot_3d_difference(polygon1, polygon2)
# Plot the difference between the two shapes angles as a heatmap
plot_heatmap_difference(polygon1, polygon2)