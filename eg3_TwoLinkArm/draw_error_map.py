import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration

def custom_formatter(x, pos):
    if np.isclose(x, 0, atol=1e-2):  # 1e-2 is the tolerance, adjust as needed
        return "0.00"
    else:
        return f"{x:.02f}"
    
def diagnosis(exp_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    figure_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    config = Configuration()
    device = config.device
    print('==> torch device: ', device)

    # Seed everything
    seed = test_settings["seed"]
    seed_everything(seed)

    # Build dynamical system
    nominal_system_name = test_settings["nominal_system_name"]
    true_system_name = test_settings["true_system_name"]
    nominal_system = get_system(nominal_system_name).to(device)
    true_system = get_system(true_system_name).to(device)
    state_space = np.array([[-np.pi, np.pi],
                            [-np.pi, np.pi],
                            [-0.1, 0.1],
                            [-0.1, 0.1]], dtype=config.np_dtype)

    # Build neural network
    nn_config = test_settings["nn_config"]
    input_bias = np.array(nn_config["input_bias"], dtype=config.np_dtype)
    input_transform = np.array(nn_config["input_transform_to_inverse"], dtype=config.np_dtype)
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

    #############################################
    N = 10
    dim1 = np.linspace(state_space[0,0], state_space[0,1], N)
    dim2 = np.linspace(state_space[1,0], state_space[1,1], N)
    dim3 = np.linspace(state_space[2,0], state_space[2,1], N)
    dim4 = np.linspace(state_space[3,0], state_space[3,1], N)

    D1, D2, D3, D4 = np.meshgrid(dim1, dim2, dim3, dim4)

    D1_flatten = D1.flatten()
    D2_flatten = D2.flatten()
    D3_flatten = D3.flatten()
    D4_flatten = D4.flatten()
    state_np = np.zeros((D1_flatten.shape[0], 4), dtype=config.np_dtype)
    state_np[:,0] = D1_flatten
    state_np[:,1] = D2_flatten
    state_np[:,2] = D3_flatten
    state_np[:,3] = D4_flatten
    state = torch.from_numpy(state_np).to(device)
    residual = model(state)
    action = torch.zeros((state.shape[0], 2), dtype=config.pt_dtype).to(device)
    selected_indices = [2, 3]
    residual_norm = torch.nn.functional.pairwise_distance(
        residual + nominal_system(state, action)[:,selected_indices], 
        true_system(state, action)[:,selected_indices]).detach().cpu().numpy()
    
    Z = residual_norm.reshape(D1.shape)
    print("==> Max Error:", np.max(Z))

    # Settings
    print("==> Drawing...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    # Draw on (q1, q2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
    N = 200
    x = np.linspace(state_space[0,0], state_space[0,1], N)
    y = np.linspace(state_space[1,0], state_space[1,1], N)
    X, Y = np.meshgrid(x, y)
    X_flatten = X.flatten()
    Y_flatten = Y.flatten()
    state_np = np.zeros((X_flatten.shape[0], 4), dtype=config.np_dtype)
    state_np[:,0] = X_flatten
    state_np[:,1] = Y_flatten
    state_np[:,2] = 0.2
    state_np[:,3] = 0.2
    state = torch.from_numpy(state_np).to(device)
    residual = model(state)
    action = torch.zeros((state.shape[0], 2), dtype=config.pt_dtype).to(device)
    selected_indices = [2, 3]
    residual_norm = torch.nn.functional.pairwise_distance(
        residual + nominal_system(state, action)[:,selected_indices], 
        true_system(state, action)[:,selected_indices]).detach().cpu().numpy()
    
    Z = residual_norm.reshape(X.shape)

    vmin = 0
    vmax = 0.5
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)

    # Add a color bar which maps values to colors.
    ticksize = 22
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.09, location='right', orientation='vertical')
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.ax.yaxis.set_major_locator(LinearLocator(5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # Set labels for the axes
    labelsize = 30
    ax.set_xlabel(r"$q1$", fontsize=labelsize, labelpad=10)
    ax.set_ylabel(r"$q2$", fontsize=labelsize, labelpad=18)
    # ax.set_zlabel(r"$\lVert f(x)-\Phi(x) \rVert$", fontsize=labelsize, labelpad=20)

    # Set ticks
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.set_yticklabels(ax.get_yticks(), 
                verticalalignment='center_baseline',
                horizontalalignment='left')
    ax.set_ylim(y.min(), y.max())
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.set_zticklabels(ax.get_zticks(), 
                verticalalignment='center',
                horizontalalignment='left')
    ax.set_zlim(vmin, vmax)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.tick_params(axis='x', which='major', labelsize=ticksize, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=ticksize, pad=0)
    ax.tick_params(axis='z', which='major', labelsize=ticksize, pad=0)

    plt.savefig(os.path.join(figure_dir, "error_map_q_{:03d}.pdf".format(exp_num)), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


    # Draw on (dq1, dq2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
    N = 200
    x = np.linspace(state_space[2,0], state_space[2,1], N)
    y = np.linspace(state_space[3,0], state_space[3,1], N)
    X, Y = np.meshgrid(x, y)
    X_flatten = X.flatten()
    Y_flatten = Y.flatten()
    state_np = np.zeros((X_flatten.shape[0], 4), dtype=config.np_dtype)
    state_np[:,2] = X_flatten
    state_np[:,3] = Y_flatten
    state = torch.from_numpy(state_np).to(device)
    residual = model(state)
    action = torch.zeros((state.shape[0], 2), dtype=config.pt_dtype).to(device)
    selected_indices = [2, 3]
    residual_norm = torch.nn.functional.pairwise_distance(
        residual + nominal_system(state, action)[:,selected_indices], 
        true_system(state, action)[:,selected_indices]).detach().cpu().numpy()
    
    Z = residual_norm.reshape(X.shape)

    vmin = 0
    vmax = 0.5
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)

    # Add a color bar which maps values to colors.
    ticksize = 22
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.09, location='right', orientation='vertical')
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.ax.yaxis.set_major_locator(LinearLocator(5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # Set labels for the axes
    labelsize = 30
    ax.set_xlabel(r"$dq1$", fontsize=labelsize, labelpad=10)
    ax.set_ylabel(r"$dq2$", fontsize=labelsize, labelpad=18)
    # ax.set_zlabel(r"$\lVert f(x)-\Phi(x) \rVert$", fontsize=labelsize, labelpad=20)

    # Set ticks
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.set_yticklabels(ax.get_yticks(), 
                verticalalignment='center_baseline',
                horizontalalignment='left')
    ax.set_ylim(y.min(), y.max())
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.set_zticklabels(ax.get_zticks(), 
                verticalalignment='center',
                horizontalalignment='left')
    ax.set_zlim(vmin, vmax)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.tick_params(axis='x', which='major', labelsize=ticksize, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=ticksize, pad=0)
    ax.tick_params(axis='z', which='major', labelsize=ticksize, pad=0)

    plt.savefig(os.path.join(figure_dir, "error_map_dq_{:03d}.pdf".format(exp_num)), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    
if __name__ == "__main__":

    exp_nums = [21, 22, 23, 24, 37, 38, 39, 40]
    for exp_num in exp_nums:
        diagnosis(exp_num)
        print("#############################################")