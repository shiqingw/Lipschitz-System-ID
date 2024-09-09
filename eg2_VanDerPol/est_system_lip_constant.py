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
from cores.dynamical_systems.systems import TwoLinkArm 
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset

def diagnosis(exp_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg2_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
    
    mu = true_system.mu
    grad_f = lambda x: np.array([[0, 1.0], [-2*mu*x[0]*x[1]-1, mu*(1-x[0]**2)]])

    # Load dataset and get trainloader
    train_config = test_settings["train_config"]
    dataset_num = int(train_config["dataset"])
    dataset_path = "eg2_VanDerPol/{:03d}/dataset.mat".format(dataset_num)
    dataset_path = "{}/datasets/{}".format(str(Path(__file__).parent.parent),dataset_path)
    dataset = DynDataset(dataset_path, config)
    print("Total data points:", len(dataset))

    L = 0
    for i in range(len(dataset)):
        if i % 10000 == 0:
            print("Processing data point:", i)
        t, x, u, x_dot = dataset[i]
        L = max(L, np.linalg.norm(grad_f(x), ord = 2))
    print("L:", L)


if __name__ == "__main__":
    for exp_num in range(1, 2):
        diagnosis(exp_num)
        