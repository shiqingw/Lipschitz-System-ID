import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt

from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset

def diagnosis(dataset_num):
    # Decide torch device
    config = Configuration()
    device = config.device
    print('==> torch device: ', device)
    
    mu = 0.02
    grad_f = lambda x: np.array([[0, 1.0], [-2*mu*x[0]*x[1]-1, mu*(1-x[0]**2)]])

    # Load dataset and get trainloader
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
    dataset_num =1
    diagnosis(dataset_num)
        