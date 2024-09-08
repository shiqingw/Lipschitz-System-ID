import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pickle

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset, get_test_and_training_data

def polygon_area(vertices):
    """
    Calculate the area of a polygon using the Shoelace Theorem.

    :param vertices: List of tuples (x, y) representing the vertices of the polygon.
    :return: The area of the polygon.
    """
    n = len(vertices)  # Number of vertices
    area = 0
    
    # Applying the Shoelace Theorem
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # The next vertex, wrap around using modulo
        area += x1 * y2 - x2 * y1
    
    return abs(area) / 2

def get_grid(dataset_num, grid_size):
    config = Configuration()
    
    dataset_folder = "{}/datasets/eg2_VanDerPol/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    dataset = DynDataset(dataset_file, config)

    x_data = []
    x_dot_data = []
    for i in range(len(dataset)):
        t, x, u, x_dot = dataset[i]
        x_data.append(x.cpu().detach().numpy())
        x_dot_data.append(x_dot.cpu().detach().numpy())
    x_data = np.array(x_data)
    x_dot_data = np.array(x_dot_data)
    
    # Step 1: Compute the convex hull of the points
    hull = ConvexHull(x_data)
    hull_points = x_data[hull.vertices]  # Vertices of the convex hull

    # Step 2: Define the grid size
    x_min, y_min = np.min(x_data, axis=0)
    x_max, y_max = np.max(x_data, axis=0)

    # Create the grid
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Step 3: Check if grid cells overlap with the convex hull
    # Create a path object for the convex hull
    hull_path = matplotlib.path.Path(hull_points)

    # List to store the grid cells that overlap with the convex hull
    selected_cells = []

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            # Define the corners of the grid cell
            grid_cell = np.array([
                [x_bins[i], y_bins[j]],
                [x_bins[i+1], y_bins[j]],
                [x_bins[i+1], y_bins[j+1]],
                [x_bins[i], y_bins[j+1]]
            ])
            # Check if any corner of the grid cell is inside the convex hull
            if hull_path.intersects_path(matplotlib.path.Path(grid_cell)):
                selected_cells.append(grid_cell)

    # Step 4: Visualize the points, the convex hull, and the selected grid cells
    fig, ax = plt.subplots()

    # Plot the 2D points
    ax.scatter(x_data[:, 0], x_data[:, 1], color='blue', label='2D Points', s=1)

    # Plot the grid cells that overlap with the convex hull
    for cell in selected_cells:
        rect = plt.Polygon(cell, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(rect)
        # print(polygon_area(cell))

    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Display the plot
    plt.legend()
    plt.savefig("{}/00_grid_size_{:.2f}.png".format(dataset_folder, grid_size))

    # Save the selected cells to a pickle file
    with open("{}/00_selected_cells_grid_size_{:.2f}.pkl".format(dataset_folder, grid_size), 'wb') as f:
        pickle.dump(np.array(selected_cells), f)


if __name__ == "__main__":
    dataset_num = 1
    grid_size = 0.1
    get_grid(dataset_num, grid_size)