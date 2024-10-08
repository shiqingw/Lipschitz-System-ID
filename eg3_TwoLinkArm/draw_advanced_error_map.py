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
from cores.dynamical_systems.systems import TwoLinkArm 
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration
from matplotlib.transforms import Bbox


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
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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

    # Build dynamical system
    nominal_system_name = test_settings["nominal_system_name"]
    true_system_name = test_settings["true_system_name"]
    nominal_system = get_system(nominal_system_name).to(device)
    true_system = get_system(true_system_name).to(device)
    state_dim = nominal_system.n_state
    control_dim = nominal_system.n_control
    state_space = np.array([np.pi, np.pi, 4.5, 4.5], dtype=config.np_dtype)
    state_space = np.vstack((-state_space, state_space)).T

    # Build neural network
    nn_config = test_settings["nn_config"]
    input_bias = np.array(nn_config["input_bias"], dtype=config.np_dtype)
    input_transform = np.array(nn_config["input_transform_to_inverse"], dtype=config.np_dtype)
    input_transform = 1.0/input_transform
    output_transform = np.array(nn_config["output_transform"], dtype=config.np_dtype)
    train_transform = bool(nn_config["train_transform"])
    zero_at_zero = bool(nn_config["zero_at_zero"])

    nn_config = get_nn_config(nn_config)
    if nn_config.layer == 'Lip_Reg':
        model = NeuralNetwork(nn_config, input_bias=None, input_transform=None, output_transform=None, train_transform=train_transform, zero_at_zero=False)
    else:
        model = NeuralNetwork(nn_config, input_bias, input_transform, output_transform, train_transform, zero_at_zero)
    if nn_config.layer == 'Sandwich':
        print("==> Lipschitz constant: {:.02f}".format(nn_config.gamma))

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_init.pt'), device)
    model.eval()
    print("==> Input transform to be applied to the neural network (init):")
    print(model.input_transform.cpu().detach().numpy())
    print("==> Output transform to be applied to the neural network (init):")
    print(model.output_transform.cpu().detach().numpy())

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()
    print("==> Input transform to be applied to the neural network (trained):")
    print(model.input_transform.cpu().detach().numpy())
    print("==> Output transform to be applied to the neural network (trained):")
    print(model.output_transform.cpu().detach().numpy())
    if nn_config.layer == 'Sandwich':
        print("==> Overall Lipschitz constant: {:.02f}".format(nn_config.gamma*\
                    max(model.input_transform.cpu().detach().numpy())*max(model.output_transform.cpu().detach().numpy())))
    
    #############################################
    # plot the error on (theta1, theta2) plane
    N = 100
    Theta1 = np.linspace(state_space[0,0], state_space[0,1], N)
    Theta2 = np.linspace(state_space[1,0], state_space[1,1], N)
    X_theta1, Y_theta2 = np.meshgrid(Theta1, Theta2)
    X_flatten = X_theta1.flatten()
    Y_flatten = Y_theta2.flatten()
    state_np = np.zeros((X_flatten.shape[0], 4), dtype=config.np_dtype)
    state_np[:,0] = X_flatten
    state_np[:,1] = Y_flatten
    state = torch.from_numpy(state_np).to(device)
    residual = model(state)
    action = torch.zeros((state.shape[0], control_dim), dtype=config.pt_dtype).to(device)
    residual_norm = torch.nn.functional.pairwise_distance(
        residual + nominal_system(state, action)[:,2:4], 
        true_system(state, action)[:,2:4]).detach().cpu().numpy()
    Z_theta = residual_norm.reshape(X_theta1.shape)


    # plot the error on (dtheta1, dtheta2) plane
    dTheta1 = np.linspace(state_space[2,0], state_space[2,1], N)
    dTheta2 = np.linspace(state_space[3,0], state_space[3,1], N)
    X_dtheta1, Y_dtheta2 = np.meshgrid(dTheta1, dTheta2)

    X_flatten = X_dtheta1.flatten()
    Y_flatten = Y_dtheta2.flatten()
    state_np = np.zeros((X_flatten.shape[0], 4), dtype=config.np_dtype)
    state_np[:,2] = X_flatten
    state_np[:,3] = Y_flatten
    state = torch.from_numpy(state_np).to(device)
    residual = model(state)
    action = torch.zeros((state.shape[0], control_dim), dtype=config.pt_dtype).to(device)
    residual_norm = torch.nn.functional.pairwise_distance(
        residual + nominal_system(state, action)[:,2:4], 
        true_system(state, action)[:,2:4]).detach().cpu().numpy()
    Z_dtheta = residual_norm.reshape(X_dtheta1.shape)


    return X_theta1, Y_theta2, Z_theta, X_dtheta1, Y_dtheta2, Z_dtheta


if __name__ == "__main__":

    exp_nums = [164, 270, 76]
    titles = ['FCNs', 'LRNs', 'Ours']
    figure_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), 0)

    Z_thetas = []
    Z_dthetas = []
    for exp_num in exp_nums:
        X_theta1, Y_theta2, Z_theta, X_dtheta1, Y_dtheta2, Z_dtheta = diagnosis(exp_num)
        Z_thetas.append(Z_theta)
        Z_dthetas.append(Z_dtheta)
        print("#############################################")
    
    print("==> Drawing...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    
    ########################## plot the error on (theta1, theta2) plane ##########################
    # 3D plot
    plot_width_3d = 18
    plot_height_3d = 7
    fig = plt.figure(figsize=(plot_width_3d, plot_height_3d))
    cells = [131, 132, 133]

    for i in range(len(exp_nums)): 
        Z  = Z_thetas[i]
        X = X_theta1
        Y = Y_theta2
        ax = fig.add_subplot(cells[i], projection='3d')
        
        vmin = 0
        vmax = 0.1
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)

        ticksize = 22
        
        # Set labels for the axes
        labelsize = 30
        ax.set_xlabel(r"$q_1$", fontsize=labelsize, labelpad=10)
        ax.set_ylabel(r"$q_2$", fontsize=labelsize, labelpad=18)

        # Set ticks
        ax.set_xlim(X.min(), X.max())
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

        ax.set_yticklabels(ax.get_yticks(), 
                    verticalalignment='center_baseline',
                    horizontalalignment='left')
        ax.set_ylim(Y.min(), Y.max())
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

        ax.set_title(titles[i], fontsize=labelsize, pad=0)

    plt.subplots_adjust(wspace=0.15)

    # Add a color bar which maps values to colors.
    cbar_width = 0.02
    cbar_hight = 0.49
    cbar_ax = fig.add_axes([0.94, (1-cbar_hight)/2, cbar_width, cbar_hight])
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.09, location='right', orientation='vertical', cax=cbar_ax)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.ax.yaxis.set_major_locator(LinearLocator(5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    exp_nums_str = "_".join([str(exp_num) for exp_num in exp_nums])
    fig_width_3d = 16
    fig_height_3d = 5
    bbox_3d = Bbox.from_bounds(2.1, (plot_height_3d - fig_height_3d)/2, fig_width_3d, fig_height_3d)
    plt.savefig(os.path.join(figure_dir, "plot_q1q2_{}_3d.pdf".format(exp_nums_str)), bbox_inches=bbox_3d)
    plt.close(fig)


    ########################## plot the error on (dtheta1, dtheta2) plane ##########################
    # 3D plot
    plot_width_3d = 18
    plot_height_3d = 7
    fig = plt.figure(figsize=(plot_width_3d, plot_height_3d))
    cells = [131, 132, 133]

    for i in range(len(exp_nums)): 
        Z  = Z_dthetas[i]
        X = X_dtheta1
        Y = Y_dtheta2
        ax = fig.add_subplot(cells[i], projection='3d')
        
        vmin = 0
        vmax = 0.5
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)

        ticksize = 22
        
        # Set labels for the axes
        labelsize = 30
        ax.set_xlabel(r"$\dot{q}_1$", fontsize=labelsize, labelpad=10)
        ax.set_ylabel(r"$\dot{q}_2$", fontsize=labelsize, labelpad=18)

        # Set ticks
        ax.set_xlim(X.min(), X.max())
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

        ax.set_yticklabels(ax.get_yticks(), 
                    verticalalignment='center_baseline',
                    horizontalalignment='left')
        ax.set_ylim(Y.min(), Y.max())
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

        ax.set_title(titles[i], fontsize=labelsize, pad=0)

    plt.subplots_adjust(wspace=0.15)

    # Add a color bar which maps values to colors.
    cbar_width = 0.02
    cbar_hight = 0.49
    cbar_ax = fig.add_axes([0.94, (1-cbar_hight)/2, cbar_width, cbar_hight])
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.09, location='right', orientation='vertical', cax=cbar_ax)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.ax.yaxis.set_major_locator(LinearLocator(5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    exp_nums_str = "_".join([str(exp_num) for exp_num in exp_nums])
    fig_width_3d = 16
    fig_height_3d = 5
    bbox_3d = Bbox.from_bounds(2.1, (plot_height_3d - fig_height_3d)/2, fig_width_3d, fig_height_3d)
    plt.savefig(os.path.join(figure_dir, "plot_dq1dq2_{}_3d.pdf".format(exp_nums_str)), bbox_inches=bbox_3d)
    plt.close(fig)





