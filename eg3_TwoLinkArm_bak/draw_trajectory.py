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

def draw(exp_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg3_results_bak/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    figure_dir = "{}/eg3_results_bak/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results_bak/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
    num_trajs = 10
    directions = np.linspace(0, 2*np.pi, num_trajs, endpoint=False)
    radii = np.linspace(0.5, 2.0, num_trajs)
    N = 1500
    dt = 0.01
    T = np.arange(0, N*dt, dt)
    X_true = np.zeros((num_trajs, N, state_dim), dtype=config.np_dtype)
    X_trained = np.zeros((num_trajs, N, state_dim), dtype=config.np_dtype)
    kp = 1.0
    kd = 2.0
    freq = 0.25
    amplitude = 25.0
    np.random.seed(0)

    for kk in range(num_trajs):
        phases = np.random.uniform(0, 2*np.pi, 2) 
        q0 = radii[kk] * np.array([np.cos(directions[kk]), np.sin(directions[kk])], dtype=config.np_dtype)
        x0 = np.concatenate((q0, np.zeros(2)), axis=0)
        q0 = torch.from_numpy(q0).to(device)

        X_true_tmp = torch.zeros((N, state_dim), dtype=config.pt_dtype)
        X_true_tmp[0,:] = torch.from_numpy(x0).to(device)
        for i in range(1, N):
            t = T[i]
            x = torch.unsqueeze(X_true_tmp[i-1,:], 0)
            gravity = true_system.gravity_vector(x).detach().cpu()
            coriolis = true_system.coriolis_vector(x).detach().cpu()
            M = true_system.mass_matrix(x).detach().cpu()
            e_q = x[:,0:2] - q0
            e_dq = x[:,2:4]
            ddq = -kp*e_q - kd*e_dq
            ddq = ddq.to(config.pt_dtype)
            control = gravity + coriolis + torch.matmul(M, torch.unsqueeze(ddq, -1)).squeeze(-1)
            time_varying = amplitude*np.sin(2*np.pi*freq*t + phases)
            control += torch.tensor(time_varying, dtype=config.pt_dtype).unsqueeze(0)
            x = x + dt*true_system(x, action=control).detach().cpu()
            X_true_tmp[i,:] = torch.squeeze(x)

        X_trained_tmp = torch.zeros((N, state_dim), dtype=config.pt_dtype)
        X_trained_tmp[0,:] = torch.from_numpy(x0).to(device)
        for i in range(1, N):
            t = T[i]
            x = torch.unsqueeze(X_trained_tmp[i-1,:], 0)
            gravity = nominal_system.gravity_vector(x).detach().cpu()
            coriolis = nominal_system.coriolis_vector(x).detach().cpu()
            M = nominal_system.mass_matrix(x).detach().cpu()
            e_q = x[:,0:2] - q0
            e_dq = x[:,2:4]
            ddq = -kp*e_q - kd*e_dq
            ddq = ddq.to(config.pt_dtype)
            control = gravity + coriolis + torch.matmul(M, torch.unsqueeze(ddq, -1)).squeeze(-1)
            time_varying = amplitude*np.sin(2*np.pi*freq*t + phases)
            control += torch.tensor(time_varying, dtype=config.pt_dtype).unsqueeze(0)
            friction = model(x).detach().cpu() # size (1, 2)
            frition = torch.cat((torch.zeros_like(friction), friction), dim=1)
            x = x + dt*(nominal_system(x, action=control).detach().cpu() + frition)
            X_trained_tmp[i,:] = torch.squeeze(x)
        
        X_true[kk,:,:] = X_true_tmp.numpy()
        X_trained[kk,:,:] = X_trained_tmp.numpy()
    
    print("==> Drawing...") 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    labelsize = 30
    ticksize = 20
    for kk in range(num_trajs):
        ax.plot(T, np.linalg.norm(X_trained[kk,:,:]-X_true[kk,:,:], 2, axis=1))
    ax.set_xlabel('Time [s]', fontsize=labelsize)
    ax.set_ylabel(r'$\lVert x - z \rVert_2$', fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.set_ylim([-0.001, 0.001])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "traj_error_wrt_t_{:03d}.pdf".format(exp_num)))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    labelsize = 30
    ticksize = 20
    for kk in range(num_trajs):
        ax.plot(T, X_true[kk,:,0], linewidth=2, linestyle='--')
        ax.plot(T, X_true[kk,:,1], linewidth=2, linestyle='--')
        ax.plot(T, X_trained[kk,:,0], linewidth=2, linestyle='-')
        ax.plot(T, X_trained[kk,:,1], linewidth=2, linestyle='-')
    ax.set_xlabel('Time [s]', fontsize=labelsize)
    ax.set_ylabel(r'$q$', fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "traj_true_q_wrt_t_{:03d}.pdf".format(exp_num)))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    labelsize = 30
    ticksize = 20
    for kk in range(num_trajs):
        ax.plot(T, X_true[kk,:,2], linewidth=2, linestyle='--')
        ax.plot(T, X_true[kk,:,3], linewidth=2, linestyle='--')
        ax.plot(T, X_trained[kk,:,2], linewidth=2, linestyle='-')
        ax.plot(T, X_trained[kk,:,3], linewidth=2, linestyle='-')
    ax.set_xlabel('Time [s]', fontsize=labelsize)
    ax.set_ylabel(r'$\dot{q}$', fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "traj_true_dq_wrt_t_{:03d}.pdf".format(exp_num)))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    labelsize = 30
    ticksize = 20
    for kk in range(num_trajs):
        ax.plot(X_true[kk,:,0], X_true[kk,:,1], linewidth=2, linestyle='-')
    ax.set_xlabel(r'$q_1$', fontsize=labelsize)
    ax.set_ylabel(r'$q_2$', fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "traj_true_q_{:03d}.pdf".format(exp_num)))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    labelsize = 30
    ticksize = 20
    for kk in range(num_trajs):
        ax.plot(X_true[kk,:,2], X_true[kk,:,3], linewidth=2, linestyle='-')
    ax.set_xlabel(r'$\dot{q}_1$', fontsize=labelsize)
    ax.set_ylabel(r'$\dot{q}_2$', fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "traj_true_dq_{:03d}.pdf".format(exp_num)))
    plt.close()

if __name__ == "__main__":

    exp_nums = [73, 74, 75, 76, 161, 162, 163, 164, 269, 270, 271, 272]
    # exp_nums = [73]
    for exp_num in exp_nums:
        draw(exp_num)
        print("##############################################")
