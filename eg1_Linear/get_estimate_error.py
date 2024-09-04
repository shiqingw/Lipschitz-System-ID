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

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset, get_test_and_training_data

def diagnosis(exp_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    config = Configuration()
    user_device = args.device
    if user_device != "None":
        config.device = torch.device(user_device)
    device = config.device
    print('==> torch device: ', device)

    # Seed everything
    seed = test_settings["seed"]
    seed_everything(seed)

    # Load dataset and get trainloader
    train_config = test_settings["train_config"]
    dataset_num = int(train_config["dataset"])
    dataset_path = "eg1_Linear/{:03d}/dataset.mat".format(dataset_num)
    dataset_path = "{}/datasets/{}".format(str(Path(__file__).parent.parent),dataset_path)
    dataset = DynDataset(dataset_path, config)

    train_ratio = train_config["train_ratio"]
    further_train_ratio = train_config["further_train_ratio"]
    _, test_dataset = get_test_and_training_data(dataset, train_ratio, 
                        further_train_ratio, seed_train_test=None, seed_actual_train=None)

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
    if nn_config.layer == 'Sandwich':
        print("==> Gamma: {:.02f}".format(nn_config.gamma))

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()
    print("==> Input transform to be applied to the neural network (trained):")
    print(model.input_transform.cpu().detach().numpy())
    print("==> Output transform to be applied to the neural network (trained):")
    print(model.output_transform.cpu().detach().numpy())
    if nn_config.layer == 'Sandwich':
        global_lipschitz = nn_config.gamma
        global_lipschitz *= max(model.input_transform.cpu().detach().numpy())
        global_lipschitz *= max(model.output_transform.cpu().detach().numpy())
    else:
        global_lipschitz = 1.0
        global_lipschitz *= max(model.input_transform.cpu().detach().numpy())
        global_lipschitz *= max(model.output_transform.cpu().detach().numpy())
        for layer in model.model:
            if isinstance(layer, torch.nn.Linear):
                global_lipschitz *= np.linalg.norm(layer.weight.cpu().detach().numpy(), 2)

    print("==> Global Lipschitz constant: {:.02f}".format(global_lipschitz))

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
    grid_size = 0.1  # This is the size of each grid cell
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

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Points, Convex Hull, and Selected Meshes')

    # Display the plot
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    for exp_num in range(1, 2):
        diagnosis(exp_num)

            