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
import pickle
import itertools

from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset


def get_grid(dataset_num, grid_size):
    config = Configuration()
    
    dataset_folder = "{}/datasets/eg1_Linear/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    dataset = DynDataset(dataset_file, config)

    x_np = dataset.x.cpu().numpy()
    x_min = np.min(x_np, axis=0)
    x_max = np.max(x_np, axis=0)
    
    dim_min = np.array([-3.0, -3.0]).astype(config.np_dtype)
    dim_max = np.array([3.0, 3.0]).astype(config.np_dtype)

    assert np.all(x_max >= dim_max)
    assert np.all(x_min <= dim_min)

    dim1_bins = np.arange(dim_min[0], dim_max[0] + grid_size, grid_size)
    dim2_bins = np.arange(dim_min[1], dim_max[1] + grid_size, grid_size)

    selected_cells = []  # Temporary list to store the batch of selected cells
    file_counter = 0     # Counter to track the number of files
    grid_counter = 0
    batch_size = int(2500000/(2**x_np.shape[1]*x_np.shape[1]))

    print("==> Preparing to save the selected grid cells in multiple pickle files")
    for i, j in itertools.product(range(len(dim1_bins) - 1), range(len(dim2_bins) - 1)):
        # Define the corners of the grid cell
        grid_cell = np.array([
            [dim1_bins[i], dim2_bins[j]],
            [dim1_bins[i+1], dim2_bins[j]],
            [dim1_bins[i+1], dim2_bins[j+1]],
            [dim1_bins[i], dim2_bins[j+1]]
        ])
        selected_cells.append(grid_cell)
        grid_counter += 1

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
    
    dataset_folder = "{}/datasets/eg1_Linear/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    dataset = DynDataset(dataset_file, config)
    x_np = dataset.x.cpu().numpy()

    rect_dim1_dim2 = []
    file_counter = 1
    while True:
        file_path = "{}/00_selected_cells_grid_size_{:.2f}_batch_{:03d}.pkl".format(dataset_folder, grid_size, file_counter)
        if not os.path.exists(file_path):
            break
        with open(file_path, 'rb') as f:
            selected_cells = pickle.load(f)
        file_counter += 1
        for cell in selected_cells:
            min1, min2 = np.min(cell, axis=0)
            max1, max2 = np.max(cell, axis=0)
            rect_dim1_dim2.append(np.array([[min1, min2], [max1, min2], [max1, max2], [min1, max2]]))
        rect_dim1_dim2 = np.array(rect_dim1_dim2)
        rect_dim1_dim2 = np.unique(rect_dim1_dim2, axis=0)
        rect_dim1_dim2 = list(rect_dim1_dim2)
    print("==> Finished loading the selected grid cells from {} files".format(file_counter - 1))
    
    fig, ax = plt.subplots()

    # Plot the 2D points
    ax.scatter(x_np[:, 0], x_np[:, 1], color='blue', label='2D Points', s=1)

    # Plot the grid cells that overlap with the convex hull
    for cell in rect_dim1_dim2:
        rect = plt.Polygon(cell, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(rect)

    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Display the plot
    plt.legend()
    plt.savefig("{}/00_grid_size_{:.2f}.png".format(dataset_folder, grid_size))

if __name__ == "__main__":
    dataset_num = 1
    grid_size = 0.1
    get_grid(dataset_num, grid_size)
    draw_grid(dataset_num, grid_size)