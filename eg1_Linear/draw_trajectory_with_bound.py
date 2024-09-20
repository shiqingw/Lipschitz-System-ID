import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import pickle

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration

def draw(exp_num, grid_sizes):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    figure_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
    x0 = np.array([[3.0, 3.0]], dtype=config.np_dtype)
    N = 1000
    dt = 0.01
    T = np.arange(0, N*dt, dt)
    X_true = np.zeros((N, state_dim), dtype=config.np_dtype)
    X_true[0,:] = x0.squeeze()

    for i in range(1, N):
        x = X_true[i-1,:][np.newaxis,:]
        x = x + dt*true_system(torch.from_numpy(x).to(device), action=torch.zeros(1)).detach().cpu().numpy()
        X_true[i,:] = x.squeeze()

    X_trained = np.zeros((N, state_dim), dtype=config.np_dtype)
    X_trained[0,:] = x0.squeeze()
    for i in range(1, N):
        x = X_trained[i-1,:][np.newaxis,:]
        x = x + dt*(nominal_system(torch.from_numpy(x).to(device), action=None).detach().cpu().numpy() +\
                    model(torch.from_numpy(x).to(device)).detach().cpu().numpy())
        X_trained[i,:] = x.squeeze()

    # Load Lipschitz bound
    lipschitz_bound = np.inf
    global_lipschitz_file = os.path.join(results_dir, "global_lipschitz.pkl")
    if os.path.exists(global_lipschitz_file):
        with open(global_lipschitz_file, "rb") as f:
            global_lipschitz_data = pickle.load(f)
            lipschitz_bound = min(lipschitz_bound, global_lipschitz_data["global_lipschitz"])
    
    lipsdp_lipschitz_file = os.path.join(results_dir, "lipsdp_lipschitz.pkl")
    if os.path.exists(lipsdp_lipschitz_file):
        with open(lipsdp_lipschitz_file, "rb") as f:
            lipsdp_lipschitz_data = pickle.load(f)
            lipschitz_bound = min(lipschitz_bound, lipsdp_lipschitz_data["lipsdp_lipschitz"])
    
    print("==> Lipschitz bound: ", lipschitz_bound)

    # Load maximum error
    max_error = np.inf
    for grid_size in grid_sizes:
        estimation_error_file = os.path.join(results_dir, "estimate_error_grid_size_{:.2f}.pkl".format(grid_size))
        if os.path.exists(estimation_error_file):
            with open(estimation_error_file, "rb") as f:
                estimation_error_data = pickle.load(f)
                # check if has the key
                if "global_lipschitz_error" in estimation_error_data:
                    max_error = min(max_error, estimation_error_data["global_lipschitz_error"])
                if "lipsdp_lipschitz_error" in estimation_error_data:
                    max_error = min(max_error, estimation_error_data["lipsdp_lipschitz_error"])
    print("==> Maximum error: ", max_error)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    plt.figure()
    plt.plot(T, np.linalg.norm(X_trained-X_true, 2, axis=1))
    plt.plot(T, max_error/lipschitz_bound*(np.exp(lipschitz_bound*T)-1), linestyle="--", label="Upper bound")
    plt.grid()
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\lVert x - z \rVert_2$')
    plt.ylim([-0.01, 0.1])
    plt.savefig(os.path.join(figure_dir, "error_wrt_t_{:03d}.pdf".format(exp_num)), bbox_inches='tight')
    plt.close()

    # clear figure
    plt.clf()
    plt.plot(X_trained[:,0], X_trained[:,1], label="trained")
    plt.plot(X_true[:,0], X_true[:,1], label="true", linestyle="--")
    plt.legend()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.axis('equal')
    plt.savefig(os.path.join(figure_dir, "traj_wrt_x_{:03d}.pdf".format(exp_num)), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    grid_sizes = [0.1, 0.05]
    exp_nums = [53, 54, 55, 56, 137, 138, 139, 140, 257, 258, 259, 260]

    for exp_num in exp_nums:
        draw(exp_num, grid_sizes)
        print("##############################################")
