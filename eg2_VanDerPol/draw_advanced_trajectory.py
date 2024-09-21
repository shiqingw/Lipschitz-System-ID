import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration

def simulate(exp_num, all_x0):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    figure_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg2_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    config = Configuration()
    device = config.device
    print('==> torch device: ', device)

    # Build dynamical system
    nominal_system_name = test_settings["nominal_system_name"]
    true_system_name = test_settings["true_system_name"]
    nominal_system = get_system(nominal_system_name).to(device)
    true_system = get_system(true_system_name).to(device)
    state_dim = nominal_system.n_state

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

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()

    # Simulate the system using euler method
    print("==> Simulating...")
    num_trajs = all_x0.shape[0]
    N = 1000
    dt = 0.01
    T = np.arange(0, N*dt, dt)
    X_true = np.zeros((num_trajs, N, state_dim), dtype=config.np_dtype)
    X_trained = np.zeros((num_trajs, N, state_dim), dtype=config.np_dtype)

    for kk in range(num_trajs):
        x0 = all_x0[kk,:]
        X_true_tmp = torch.zeros((N, state_dim), dtype=config.pt_dtype)
        X_true_tmp[0,:] = torch.from_numpy(x0).to(device)
        for i in range(1, N):
            x = torch.unsqueeze(X_true_tmp[i-1,:], 0)
            x = x + dt*true_system(x, action=torch.zeros(1)).detach().cpu()
            X_true_tmp[i,:] = torch.squeeze(x)

        X_trained_tmp = torch.zeros((N, state_dim), dtype=config.pt_dtype)
        X_trained_tmp[0,:] = torch.from_numpy(x0).to(device)
        for i in range(1, N):
            x = torch.unsqueeze(X_trained_tmp[i-1,:], 0)
            x = x + dt*(nominal_system(x, action=None).detach().cpu() +\
                        model(x).detach().cpu())
            X_trained_tmp[i,:] = torch.squeeze(x)
        
        X_true[kk,:,:] = X_true_tmp.numpy()
        X_trained[kk,:,:] = X_trained_tmp.numpy()

    return T, X_true, X_trained
    

if __name__ == "__main__":

    exp_nums = [142, 236, 62]
    labels = ['FCNs', 'LRNs', 'Ours']
    colors = ['tab:orange', 'tab:green', 'tab:blue']

    np.random.seed(0)
    num_trajs = 100
    all_x0 = np.random.uniform(-1, 1, (num_trajs, 2))
    state_bounds = np.array([2.5,2.5])
    all_x0 = all_x0 * state_bounds

    print("==> Drawing...") 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fig_all, ax_all = plt.subplots(figsize=(8, 6))
    for (i, exp_num) in enumerate(exp_nums):
        label_name = labels[i]
        color = colors[i]
        T, X_true, X_trained = simulate(exp_num, all_x0)
        mean = np.mean(np.linalg.norm(X_trained-X_true, 2, axis=2), axis=0)
        std = np.std(np.linalg.norm(X_trained-X_true, 2, axis=2), axis=0)

        # Draw in the collective figure
        ax_all.plot(T, mean, linestyle='-', linewidth=2, label=label_name, color=color)
        ax_all.fill_between(T, mean-std, mean+std, color=color, alpha=0.5)

        # Draw in the individual figure
        figure_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        labelsize = 30
        ticksize = 20
        for kk in range(num_trajs):
            ax.plot(T, np.linalg.norm(X_trained[kk,:,:]-X_true[kk,:,:], 2, axis=1))
        ax.plot(T, mean, 'k', linestyle='--', linewidth=3)
        ax.fill_between(T, mean-std, mean+std, color='gray', alpha=0.5)
        ax.set_xlabel('Time [s]', fontsize=labelsize)
        ax.set_ylabel(r'$\lVert x - z \rVert_2$', fontsize=labelsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "plot_traj_error_wrt_t_{:03d}.pdf".format(exp_num)))
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        labelsize = 30
        ticksize = 20
        for kk in range(num_trajs):
            ax.plot(X_true[kk,:,0], X_true[kk,:,1],linewidth=2)
        ax.set_xlabel(r'$x_1$', fontsize=labelsize)
        ax.set_ylabel(r'$x_2$', fontsize=labelsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, "plot_traj_true_{:03d}.pdf".format(exp_num)))
        plt.close()
    
    labelsize = 25
    ticksize = 25
    ax_all.set_xlabel('Time [s]', fontsize=labelsize)
    ax_all.set_ylabel(r'$\lVert z(t) - x(t) \rVert_2$', fontsize=labelsize)
    ax_all.tick_params(axis='both', which='major', labelsize=ticksize)
    ax_all.set_ylim([-0.0016, 0.016])
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize=ticksize)
    exp_nums_str = "_".join([str(exp_num) for exp_num in exp_nums])
    plt.savefig(os.path.join(figure_dir, "plot_traj_error_wrt_t_{}.pdf".format(exp_nums_str)))
        
    
