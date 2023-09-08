import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import translate, rotate

# Generate a random polygon
def random_polygon(vertices_count):
    angles = np.sort(2 * np.pi * np.random.rand(vertices_count))
    radii = 1 + np.random.rand(vertices_count) * 2
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return list(zip(x, y))

# Compute the histogram representation of the polygon
def polygon_histogram(polygon):
    rotations = []
    cumulative_lengths = [0]
    for i in range(len(polygon)):
        x1, y1 = polygon[i - 1]
        x2, y2 = polygon[i]
        
        edge_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        cumulative_lengths.append(cumulative_lengths[-1] + edge_length)
        
        angle = np.arctan2(y2 - y1, x2 - x1)
        if i == 0:
            prev_angle = angle
        rotation = angle - prev_angle
        rotations.append(rotation)
        prev_angle = angle
    return cumulative_lengths[1:], rotations

# Define the MSE function for the optimization
def mse(shift, hist1, hist2):
    shifted_hist2 = [x + shift for x in hist2[0]]
    mse_value = 0
    for x1, y1 in zip(hist1[0], hist1[1]):
        min_error = float('inf')
        for x2, y2 in zip(shifted_hist2, hist2[1]):
            error = (y1 - y2)**2 + (x1 - x2)**2
            if error < min_error:
                min_error = error
        mse_value += min_error
    return mse_value / len(hist1[0])

# Adjusted MSE function to find valleys with added vertical and horizontal shifts
def valley_mse(params, hist1, hist2):
    slide, dx, dy = params
    shifted_hist2_x = [x + slide for x in hist2[0]]
    shifted_hist2_y = [y + dy for y in hist2[1]]
    total_height = 0

    for x1, y1 in zip(hist1[0], hist1[1]):
        x1 += dx
        y1 += dy
        for x2, y2 in zip(shifted_hist2_x, shifted_hist2_y):
            if x2 <= x1 <= x2 or x2 <= x1 <= x2:
                total_height += abs(y1 + y2)  # Using absolute value since we're dealing with both directions now

    return total_height

# Generate random polygons
poly1 = random_polygon(6)
poly2 = random_polygon(6)

# Compute histograms
hist1 = polygon_histogram(poly1)
hist2 = polygon_histogram(poly2)

# Find optimal shift
initial_guess = [0, 0, 0]  # slide, dx, dy
optimal_params = fmin(valley_mse, initial_guess, args=(hist1, hist2), disp=False)

# Extract the optimal slide, dx, dy
optimal_slide, optimal_dx, optimal_dy = optimal_params

# Plot histograms and polygons
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Histogram for Polygon 1
start_x = 0
for x_width, y_height in zip(hist1[0], hist1[1]):
    axes[0].bar(start_x, y_height, width=x_width, align='edge', alpha=0.7)
    start_x += x_width

# Histogram for Polygon 2 (shifted based on optimal_slide)
start_x = optimal_slide
for x_width, y_height in zip(hist2[0], hist2[1]):
    axes[0].bar(start_x, y_height, width=x_width, align='edge', alpha=0.7, color='orange')
    start_x += x_width

axes[0].set_title('Histograms')

# Plotting the polygons
polygon1 = translate(Polygon(poly1), xoff=optimal_dx, yoff=optimal_dy)
polygon2 = translate(Polygon(poly2), xoff=optimal_dx, yoff=optimal_dy)

axes[1].plot(*polygon1.exterior.xy, label="Polygon 1 (shifted)")
axes[1].plot(*polygon2.exterior.xy, label="Polygon 2 (shifted)")
axes[1].legend()
axes[1].set_title('Polygons')

plt.tight_layout()
plt.show()