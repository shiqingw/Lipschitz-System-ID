import json
import sys
import os
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt

from cores.dynamical_systems.create_system import get_system
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset, get_test_and_training_data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    figure_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)

    # Load test settings
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    config = Configuration()
    user_device = args.device
    if user_device != "None":
        config.device = torch.device(user_device)
    device = config.device
    print('==> torch device: ', device)

    # Build dynamical system
    system_name = test_settings["nominal_system_name"]
    system = get_system(system_name).to(device)

    # Load dataset and get trainloader
    train_config = test_settings["train_config"]
    dataset_num = int(train_config["dataset"])
    dataset_path = "eg3_TwoLinkArm/{:03d}/dataset.mat".format(dataset_num)
    dataset_path = "{}/datasets/{}".format(str(Path(__file__).parent.parent),dataset_path)
    dataset = DynDataset(dataset_path, config)
    print("Total data points:", len(dataset))

    t, x, u, x_dot = dataset[:]
    print("Input mean:", torch.mean(x, dim=0))
    print("Input std:", torch.std(x, dim=0))
    print("Target mean:", torch.mean(x_dot - system(x, u), dim=0))
    print("Target std:", torch.std(x_dot - system(x, u), dim=0))
    print("Target max:", torch.max(torch.abs(x_dot - system(x, u)), dim=0))

    train_ratio = train_config["train_ratio"]
    further_train_ratio = train_config["further_train_ratio"]
    seed_train_test = train_config["seed_train_test"]
    seed_actual_train = train_config["seed_actual_train"]
    train_dataset, test_dataset = get_test_and_training_data(dataset, train_ratio, further_train_ratio, seed_train_test, seed_actual_train)

    # draw train and test data
    print("==> Drawing...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    # (q1, q2)
    fig, ax = plt.subplots(figsize=(8,6))
    t, x, u, x_dot = train_dataset[:]
    plt.scatter(x[:,0], x[:,1], label="train", s=0.5, zorder=2.0)
    t, x, u, x_dot = test_dataset[:]
    plt.scatter(x[:,0], x[:,1], label="test", s=0.5, zorder=2.1)
    # plt.axis('equal')
    plt.grid(zorder=0.5)

    labelsize = 40
    ticksize = 30
    ax.set_xlabel(r"$q_1$", fontsize=labelsize)
    ax.set_ylabel(r"$q_2$", fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)

    # plt.legend(markerscale = 5, fontsize = labelsize,  loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "data_{:03d}_q1q2.png".format(dataset_num)), dpi=100)
    plt.show()


    # (dq1, dq2) 
    fig, ax = plt.subplots(figsize=(8,6))
    t, x, u, x_dot = train_dataset[:]
    plt.scatter(x[:,2], x[:,3], label="train", s=0.5, zorder=2.0)
    t, x, u, x_dot = test_dataset[:]
    plt.scatter(x[:,2], x[:,3], label="test", s=0.5, zorder=2.1)
    # plt.axis('equal')
    plt.grid(zorder=0.5)

    labelsize = 40
    ticksize = 30
    ax.set_xlabel(r"$\dot{q}_1$", fontsize=labelsize)
    ax.set_ylabel(r"$\dot{q}_2$", fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)

    # plt.legend(markerscale = 5, fontsize = labelsize,  loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "data_{:03d}_dq1dq2.png".format(dataset_num)), dpi=100)
    plt.show()