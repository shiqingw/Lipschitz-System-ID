import numpy as np
from scipy.spatial import KDTree

# Generate some random 2D data points
data_points = np.array([[1,1,1,1],
                       [0,0,0,0]])

# Create a KDTree from the data points
kd_tree = KDTree(data_points)

# Query the nearest neighbor of a new point
query_point = np.array([0.5, 0.5, 0.5, 0.5])
distance, index = kd_tree.query(query_point, k=2)
print(kd_tree.query_ball_point(query_point, 0.4, p=np.inf))

# Print the nearest neighbor and its distance
print(f"Nearest neighbor index: {index}")
print(f"Nearest neighbor coordinates: {data_points[index]}")
print(f"Distance to nearest neighbor: {distance}")