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
from scipy.spatial import ConvexHull, Delaunay
import pickle
from matplotlib.patches import Polygon

from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset
import itertools

def get_grid(dataset_num, grid_size, batch_size=100):
    config = Configuration()
    
    dataset_folder = "{}/datasets/eg3_TwoLinkArm/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    dataset = DynDataset(dataset_file, config)

    x_data = []
    for i in range(len(dataset)):
        t, x, u, x_dot = dataset[i]
        x_data.append(x.cpu().detach().numpy())
    x_data = np.array(x_data)

    # Step 1: Compute the convex hull of the points
    hull = ConvexHull(x_data)
    delaunay = Delaunay(hull.points[hull.vertices])

    # Step 2: Define the grid size
    x1_min, x2_min, x3_min, x4_min = np.min(x_data, axis=0)
    x1_max, x2_max, x3_max, x4_max = np.max(x_data, axis=0)

    # Create the grid
    x1_bins = np.arange(x1_min, x1_max + 2*grid_size, grid_size)
    x2_bins = np.arange(x2_min, x2_max + 2*grid_size, grid_size)
    x3_bins = np.arange(x3_min, x3_max + 2*grid_size, grid_size)
    x4_bins = np.arange(x4_min, x4_max + 2*grid_size, grid_size)

    selected_cells = []  # Temporary list to store the batch of selected cells
    file_counter = 0     # Counter to track the number of files
    grid_counter = 0     # Counter to track the number of selected grid cells

    print("==> Preparing to save the selected grid cells in multiple pickle files")
    for i, j, k, l in itertools.product(range(len(x1_bins) - 1), range(len(x2_bins) - 1), range(len(x3_bins) - 1), range(len(x4_bins) - 1)):
        # Define the corners of the grid cell
        grid_cell = np.array([
            [x1_bins[i], x2_bins[j], x3_bins[k], x4_bins[l]],
            [x1_bins[i+1], x2_bins[j], x3_bins[k], x4_bins[l]],
            [x1_bins[i], x2_bins[j+1], x3_bins[k], x4_bins[l]],
            [x1_bins[i], x2_bins[j], x3_bins[k+1], x4_bins[l]],
            [x1_bins[i], x2_bins[j], x3_bins[k], x4_bins[l+1]],
            [x1_bins[i+1], x2_bins[j+1], x3_bins[k], x4_bins[l]],
            [x1_bins[i+1], x2_bins[j], x3_bins[k+1], x4_bins[l]],
            [x1_bins[i+1], x2_bins[j], x3_bins[k], x4_bins[l+1]],
            [x1_bins[i], x2_bins[j+1], x3_bins[k+1], x4_bins[l]],
            [x1_bins[i], x2_bins[j+1], x3_bins[k], x4_bins[l+1]],
            [x1_bins[i], x2_bins[j], x3_bins[k+1], x4_bins[l+1]],
            [x1_bins[i], x2_bins[j+1], x3_bins[k+1], x4_bins[l+1]],
            [x1_bins[i+1], x2_bins[j], x3_bins[k+1], x4_bins[l+1]],
            [x1_bins[i+1], x2_bins[j+1], x3_bins[k], x4_bins[l+1]],
            [x1_bins[i+1], x2_bins[j+1], x3_bins[k+1], x4_bins[l]],
            [x1_bins[i+1], x2_bins[j+1], x3_bins[k+1], x4_bins[l+1]]
        ])

        # Check if any corner of the grid cell is inside the convex hull
        if np.any(delaunay.find_simplex(grid_cell) >= 0):
            selected_cells.append(grid_cell)
            grid_counter += 1

        # Save cells in batches to a new file
        if len(selected_cells) >= batch_size:
            file_counter += 1  # Increment the file counter
            file_path = "{}/00_selected_cells_grid_size_{:.2f}_batch_{:03d}.pkl".format(dataset_folder, grid_size, file_counter)
            with open(file_path, 'wb') as f:
                pickle.dump(np.array(selected_cells), f)
            selected_cells.clear()  # Clear the batch after saving

            print("==> Saved {} bacthes of selected grid cells".format(file_counter))

    # Save any remaining cells after the loop
    if len(selected_cells) > 0:
        file_counter += 1  # Increment the file counter
        file_path = "{}/00_selected_cells_grid_size_{:.2f}_batch_{:03d}.pkl".format(dataset_folder, grid_size, file_counter)
        with open(file_path, 'wb') as f:
            pickle.dump(np.array(selected_cells), f)
        selected_cells.clear()  # Clear the remaining batch

        print("==> Saved {} bacthes of selected grid cells".format(file_counter))
    
    print("==> Total number of selected grid cells: {}".format(grid_counter))
    print("==> Total number of files saved: {}".format(file_counter))
    
    print("==> Completed saving all selected grid cells in multiple files.")

def draw_grid(dataset_num, grid_size):
    print("==> Drawing the grid cells on the dataset")
    config = Configuration()
    
    dataset_folder = "{}/datasets/eg3_TwoLinkArm/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    dataset = DynDataset(dataset_file, config)

    x_data = []
    for i in range(len(dataset)):
        t, x, u, x_dot = dataset[i]
        x_data.append(x.cpu().detach().numpy())
    x_data = np.array(x_data)

    # Load the selected grid cells from multiple pickle files
    rect_dim1_dim2 = []
    rect_dim3_dim4 = []
    file_counter = 1
    while True:
        file_path = "{}/00_selected_cells_grid_size_{:.2f}_batch_{:03d}.pkl".format(dataset_folder, grid_size, file_counter)
        if not os.path.exists(file_path):
            break
        with open(file_path, 'rb') as f:
            selected_cells = pickle.load(f)
        file_counter += 1
        for cell in selected_cells:
            min1, min2, min3, min4 = np.min(cell, axis=0)
            max1, max2, max3, max4 = np.max(cell, axis=0)
            rect_dim1_dim2.append(np.array([[min1, min2], [max1, min2], [max1, max2], [min1, max2]]))
            rect_dim3_dim4.append(np.array([[min3, min4], [max3, min4], [max3, max4], [min3, max4]]))
        rect_dim1_dim2 = np.array(rect_dim1_dim2)
        rect_dim3_dim4 = np.array(rect_dim3_dim4)
        rect_dim1_dim2 = np.unique(rect_dim1_dim2, axis=0)
        rect_dim3_dim4 = np.unique(rect_dim3_dim4, axis=0)
        rect_dim1_dim2 = list(rect_dim1_dim2)
        rect_dim3_dim4 = list(rect_dim3_dim4)
        print("==> Loaded selected grid cells from file: {}".format(file_path))
    print("==> Finished loading the selected grid cells from multiple files")
    # First figure: Projection on Dimensions 1 and 2
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the points on the first two dimensions
    ax.scatter(x_data[:, 0], x_data[:, 1], color='blue', label='2D Points', s=0.5)

    # Plot the grid cells that overlap with the convex hull
    for rect in rect_dim1_dim2:
        rect = Polygon(rect, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(rect)

    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_title('Projection on Dimensions 1 and 2')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # Save the first figure
    plt.savefig(f"{dataset_folder}/00_grid_size_{grid_size:.2f}_dim1_dim2.png")
    plt.close(fig)  # Close the figure to free up memory
    print("==> Saved the first figure")

    # Second figure: Projection on Dimensions 3 and 4
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the points on the last two dimensions
    ax.scatter(x_data[:, 2], x_data[:, 3], color='blue', label='2D Points', s=0.5)

    # Plot the grid cells that overlap with the convex hull
    for rect in rect_dim3_dim4:
        rect = Polygon(rect, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(rect)

    ax.set_xlabel('dq1')
    ax.set_ylabel('dq2')
    ax.set_title('Projection on Dimensions 3 and 4')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # Save the second figure
    plt.savefig(f"{dataset_folder}/00_grid_size_{grid_size:.2f}_dim3_dim4.png")
    plt.close(fig)  # Close the figure to free up memory
    print("==> Saved the second figure")

if __name__ == "__main__":
    dataset_num = 1
    grid_size = 0.1
    batch_size = 10000
    # get_grid(dataset_num, grid_size, batch_size)
    draw_grid(dataset_num, grid_size)