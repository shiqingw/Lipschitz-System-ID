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
from scipy.spatial import KDTree

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import get_nn_config, load_nn_weights, format_time
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset

def find_points_in_or_near_cell(cell, kd_tree, grid_size):
    cell_center = np.mean(cell, axis=0)
    indices_point_in_cell = kd_tree.query_ball_point(cell_center, grid_size/2, p=np.inf)
    if len(indices_point_in_cell) > 0:
        indices_point_to_cell = []
    else:
        _, indices_point_to_cell = kd_tree.query(cell_center, k = 10, p=2)
    return indices_point_in_cell, indices_point_to_cell

def estimate_error(exp_num, system_lipschitz, dataset, x, kd_tree, dataset_folder, grid_size):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg2_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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

    # Build dynamical system
    system_name = test_settings["nominal_system_name"]
    nominal_system = get_system(system_name).to(device)

    selected_indices = [0,1]
    x_torch = dataset.x
    u_torch = dataset.u
    x_dot = dataset.x_dot[:,selected_indices].cpu().detach().numpy()
    x_dot_pred = model(x_torch).cpu().detach().numpy() + nominal_system(x_torch, u_torch)[:,selected_indices].cpu().detach().numpy()
    pred_error = np.linalg.norm(x_dot_pred-x_dot, 2, axis=1)

    # keep_ind = np.where((x[:,0] >= -4.0) & (x[:,0] <= 4.0) & (x[:,1] >= -4.0) & (x[:,1] <= 4.0))[0]
    # print(np.max(pred_error[keep_ind]))
    # assert False

    data = {}

    # global lipschitz error
    global_lipschitz_file  = "{}/global_lipschitz.pkl".format(results_dir)
    # check if the file exists
    if os.path.exists(global_lipschitz_file):
        print("==> Global lipschitz file exists")
        # load the global lipschitz constant
        with open(global_lipschitz_file, "rb") as f:
            global_lipschitz_data = pickle.load(f)
        global_lipschitz = global_lipschitz_data["global_lipschitz"]
        global_lipschitz_error = 0.0
        file_counter = 1
        time_start = time.time()
        while True:
            file_path = "{}/00_selected_cells_grid_size_{:.2f}_batch_{:03d}.pkl".format(dataset_folder, grid_size, file_counter)
            if not os.path.exists(file_path):
                break
            with open(file_path, 'rb') as f:
                selected_cells = pickle.load(f)
            for i in range(len(selected_cells)):
                cell_coords = selected_cells[i]
                points_in_cells_indices, closest_points_indices = find_points_in_or_near_cell(cell_coords, kd_tree, grid_size)
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
                global_lipschitz_error = max(global_lipschitz_error, min(errors))
            print("Treated {} files. global_lipschitz_error so far: {:.03f}. Time per file {:.02f} seconds.".format(file_counter,
                     global_lipschitz_error, (time.time() - time_start)/file_counter))
            file_counter += 1
        time_end = time.time()
        print("==> Max error using global_lipschitz: {:.02f}".format(global_lipschitz_error))
        data["global_lipschitz_error"] = global_lipschitz_error
        data["global_lipschitz_estimation_time"] = time_end - time_start

    # lipsdp lipschitz error
    lipsdp_lipschitz_file  = "{}/lipsdp_lipschitz.pkl".format(results_dir)
    # check if the file exists
    if os.path.exists(lipsdp_lipschitz_file):
        print("==> Lipsdp lipschitz file exists")
        # load the global lipschitz constant
        with open(lipsdp_lipschitz_file, "rb") as f:
            lipsdp_lipschitz_data = pickle.load(f)
        lipsdp_lipschitz = lipsdp_lipschitz_data["lipsdp_lipschitz"]
        lipsdp_lipschitz_error = 0.0
        file_counter = 1
        time_start = time.time()
        while True:
            file_path = "{}/00_selected_cells_grid_size_{:.2f}_batch_{:03d}.pkl".format(dataset_folder, grid_size, file_counter)
            if not os.path.exists(file_path):
                break
            with open(file_path, 'rb') as f:
                selected_cells = pickle.load(f)
            for i in range(len(selected_cells)):
                cell_coords = selected_cells[i]
                points_in_cells_indices, closest_points_indices = find_points_in_or_near_cell(cell_coords, kd_tree, grid_size)
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
                lipsdp_lipschitz_error = max(lipsdp_lipschitz_error, min(errors))
            print("Treated {} files. lipsdp_lipschitz_error so far: {:.03f}. Time per file {:.02f} seconds.".format(file_counter, \
                            lipsdp_lipschitz_error, (time.time() - time_start)/file_counter))
            file_counter += 1
        time_end = time.time()
        print("==> Max error using lipsdp_lipschitz: {:.02f}".format(lipsdp_lipschitz_error))
        data["lipsdp_lipschitz_error"] = lipsdp_lipschitz_error
        data["lipsdp_lipschitz_estimation_time"] = time_end - time_start
    
    with open("{}/estimate_error_grid_size_{:.2f}.pkl".format(results_dir, grid_size), "wb") as f:
        pickle.dump(data, f)
                
if __name__ == "__main__":
    dataset_num = 1
    grid_sizes = [0.1, 0.05]
    # exp_nums = [77, 78, 79, 80, 161, 162, 163, 164]
    exp_nums = [65]
    
    dataset_folder = "{}/datasets/eg2_VanDerPol/{:03d}".format(str(Path(__file__).parent.parent), dataset_num)
    dataset_file = "{}/dataset.mat".format(dataset_folder)
    config = Configuration()
    dataset = DynDataset(dataset_file, config)
    x = dataset.x.cpu().detach().numpy()
    kd_tree = KDTree(x)

    system_lipschitz = 1.96

    for grid_size in grid_sizes:

        for exp_num in exp_nums:
            time_start = time.time()
            estimate_error(exp_num, system_lipschitz, dataset, x, kd_tree, dataset_folder, grid_size)
            time_end = time.time()
            print("==> Time taken: {}".format(format_time(time_end - time_start)))
            print("##############################################")
            

            