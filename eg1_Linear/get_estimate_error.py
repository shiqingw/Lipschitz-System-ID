import os
import sys
import torch
import time
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance

from cores.lip_nn.models import NeuralNetwork
from cores.utils.utils import get_nn_config, load_nn_weights
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset

def find_points_in_or_near_cells_presorted(x, sorted_indices_x, sorted_x_in_x, sorted_indices_y, sorted_x_in_y, selected_cells):
    points_in_cells_idx = []
    closest_points_idx = []
    
    # Iterate over each cell
    for cell in selected_cells:
        # Extract the min and max x and y coordinates of the axis-aligned cell
        min_x, max_x = np.min(cell[:, 0]), np.max(cell[:, 0])
        min_y, max_y = np.min(cell[:, 1]), np.max(cell[:, 1])
        
        # Binary search to find the range of x coordinates in sorted_x within [min_x, max_x]
        x_min_idx = np.searchsorted(sorted_x_in_x[:, 0], min_x, side='left')
        x_max_idx = np.searchsorted(sorted_x_in_x[:, 0], max_x, side='right')
        x_min_idx = np.clip(x_min_idx, 0, len(sorted_indices_x) - 1)
        x_max_idx = np.clip(x_max_idx, 0, len(sorted_indices_x) - 1)
        x_range_indices = sorted_indices_x[x_min_idx:x_max_idx]

        # Binary search to find the range of y coordinates in sorted_y within [min_y, max_y]
        y_min_idx = np.searchsorted(sorted_x_in_y[:, 1], min_y, side='left')
        y_max_idx = np.searchsorted(sorted_x_in_y[:, 1], max_y, side='right')
        y_min_idx = np.clip(y_min_idx, 0, len(sorted_indices_y) - 1)
        y_max_idx = np.clip(y_max_idx, 0, len(sorted_indices_y) - 1)
        y_range_indices = sorted_indices_y[y_min_idx:y_max_idx]

        # Find the points that satisfy both x and y range conditions
        valid_indices = np.intersect1d(x_range_indices, y_range_indices)

        if len(valid_indices) > 0:
            points_in_cells_idx.append(valid_indices)
            closest_points_idx.append([])  # No need to find the closest point since points are inside
        else:
            # If no points are inside, find the closest point to the cell
            cell_center = np.mean(cell, axis=0)  # Approximate the cell center as its mean
            dists = distance.cdist([cell_center], x).flatten()  # Compute distances from the cell center to all points in x
            closest_points_indices = np.argsort(dists)[:4]
            points_in_cells_idx.append([])  # No points inside the cell
            closest_points_idx.append([*closest_points_indices])  # Closest point to the cell
    
    return points_in_cells_idx, closest_points_idx

def estimate_error(exp_num, system_lipschitz, dataset, selected_cells, x, sorted_indices_x, sorted_x_in_x, sorted_indices_y, sorted_x_in_y, grid_size):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    config = Configuration()
    device = config.device
    print('==> torch device: ', device)

    # Build neural network
    nn_config = test_settings["nn_config"]
    input_bias = np.array(nn_config["input_bias"], dtype=config.np_dtype)
    input_transform = np.array(nn_config["input_transform_to_inverse"], dtype=config.np_dtype)
    input_transform = np.zeros_like(input_transform)
    output_transform = np.array(nn_config["output_transform"], dtype=config.np_dtype)
    train_transform = bool(nn_config["train_transform"])
    zero_at_zero = bool(nn_config["zero_at_zero"])

    nn_config = get_nn_config(nn_config)
    if nn_config.layer == 'Lip_Reg':
        model = NeuralNetwork(nn_config, input_bias=None, input_transform=None, output_transform=None, train_transform=train_transform, zero_at_zero=False)
    else:
        model = NeuralNetwork(nn_config, input_bias, input_transform, output_transform, train_transform, zero_at_zero)

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()

    x_torch = dataset.x
    x_dot = dataset.x_dot.cpu().detach().numpy()
    x_dot_pred = model(x_torch).cpu().detach().numpy()
    pred_error = np.linalg.norm(x_dot_pred-x_dot, 2, axis=1)
    global_lipschitz_error_per_cell = np.zeros(len(selected_cells))
    lipsdp_lipschitz_error_per_cell = np.zeros(len(selected_cells))

    points_in_cells_idx, closest_points_idx = find_points_in_or_near_cells_presorted(x, sorted_indices_x, sorted_x_in_x, sorted_indices_y, sorted_x_in_y, selected_cells)

    data = {}
    global_lipschitz_file  = "{}/global_lipschitz.pkl".format(results_dir)
    # check if the file exists
    if os.path.exists(global_lipschitz_file):
        # load the global lipschitz constant
        with open(global_lipschitz_file, "rb") as f:
            global_lipschitz_data = pickle.load(f)
        global_lipschitz = global_lipschitz_data["global_lipschitz"]
        # compute the error for each cell
        for i in range(len(selected_cells)):
            cell_coords = selected_cells[i]
            points_in_cells_indices = points_in_cells_idx[i]
            closest_points_indices = closest_points_idx[i]
            errors = []
            if len(points_in_cells_indices) > 0:
                for idx in points_in_cells_indices:
                    dist = max(np.linalg.norm(cell_coords - x[idx], 2, axis=1))
                    error = pred_error[idx] + (system_lipschitz + global_lipschitz) * dist
                    errors.append(error)
            else:
                for idx in closest_points_indices:
                    dist = max(np.linalg.norm(cell_coords - x[idx], 2, axis=1))
                    error = pred_error[idx] + (system_lipschitz + global_lipschitz) * dist
                    errors.append(error)
            global_lipschitz_error_per_cell[i] = min(errors)
        print("==> Max error using global_lipschitz: {:.02f}".format(max(global_lipschitz_error_per_cell)))
        data["global_lipschitz_error"] = max(global_lipschitz_error_per_cell)
    
    lipsdp_lipschitz_file  = "{}/lipsdp_lipschitz.pkl".format(results_dir)
    # check if the file exists
    if os.path.exists(lipsdp_lipschitz_file):
        # load the lipsdp lipschitz constant
        with open(lipsdp_lipschitz_file, "rb") as f:
            lipsdp_lipschitz_data = pickle.load(f)
        lipsdp_lipschitz = lipsdp_lipschitz_data["lipsdp_lipschitz"]
        # compute the error for each cell
        for i in range(len(selected_cells)):
            cell_coords = selected_cells[i]
            points_in_cells_indices = points_in_cells_idx[i]
            closest_points_indices = closest_points_idx[i]
            errors = []
            if len(points_in_cells_indices) > 0:
                for idx in points_in_cells_indices:
                    dist = max(np.linalg.norm(cell_coords - x[idx], 2, axis=1))
                    error = pred_error[idx] + (system_lipschitz + lipsdp_lipschitz) * dist
                    errors.append(error)
            else:
                for idx in closest_points_indices:
                    dist = max(np.linalg.norm(cell_coords - x[idx], 2, axis=1))
                    error = pred_error[idx] + (system_lipschitz + lipsdp_lipschitz) * dist
                    errors.append(error)
            lipsdp_lipschitz_error_per_cell[i] = min(errors)
        print("==> Max error using lipsdp_lipschitz: {:.02f}".format(max(lipsdp_lipschitz_error_per_cell)))
        data["lipsdp_lipschitz_error"] = max(lipsdp_lipschitz_error_per_cell)
    
    with open("{}/estimate_error_grid_size_{:.2f}.pkl".format(results_dir, grid_size), "wb") as f:
        pickle.dump(data, f)
                
if __name__ == "__main__":
    dataset_num = 1
    grid_sizes = [0.5, 0.25, 0.1]
    exp_nums = [1, 67, 68, 69, 2, 70, 71, 72, 3, 73, 74, 75, 4, 76, 77, 78, 5, 79, 80, 81, 6, 82, 83, 84] + \
                [7, 85, 86, 87, 8, 88, 89, 90, 9, 91, 92, 93, 10, 94, 95, 96, 11, 97, 98, 99, 12, 100, 101, 102] + \
                [13, 103, 104, 105, 14, 106, 107, 108, 15, 109, 110, 111, 16, 112, 113, 114, 17, 115, 116, 117, 18, 118, 119, 120] + \
                [19, 121, 122, 123, 29, 151, 152, 153, 36, 172, 173, 174] + \
                [49, 211, 212, 213, 57, 235, 236, 237, 65, 259, 260, 261]

    for grid_size in grid_sizes:
        dataset_folder = "{}/datasets/eg1_Linear/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
        dataset_file = "{}/dataset.mat".format(dataset_folder)
        config = Configuration()
        dataset = DynDataset(dataset_file, config)
        with open("{}/00_selected_cells_grid_size_{:.2f}.pkl".format(dataset_folder, grid_size), "rb") as f:
            selected_cells = pickle.load(f)

        x = dataset.x.cpu().detach().numpy()
        
        sorted_indices_x = np.argsort(x[:, 0])
        sorted_x_in_x = x[sorted_indices_x]

        sorted_indices_y = np.argsort(x[:, 1])
        sorted_x_in_y = x[sorted_indices_y]

        system_lipschitz = np.linalg.norm(np.array([[-0.1, 2.0], [-2.0, -0.1]]), 2)

        for exp_num in exp_nums:
            time_start = time.time()
            estimate_error(exp_num, system_lipschitz, dataset, selected_cells, x, sorted_indices_x, sorted_x_in_x, sorted_indices_y, sorted_x_in_y, grid_size)
            time_end = time.time()
            print("==> Time taken: {:.02f} seconds".format(time_end - time_start))
            print("##############################################")

            