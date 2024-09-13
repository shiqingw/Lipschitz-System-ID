import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pickle

from cores.lip_nn.models import NeuralNetwork
from cores.utils.utils import get_nn_config, load_nn_weights
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset


def global_lipschitz(exp_num):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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

    print("Global Lipschitz constant: {:.02f}".format(global_lipschitz))

    # Save the global Lipschitz constant
    print("==> Saving to file ...")
    data = {'global_lipschitz': global_lipschitz}
    with open("{}/global_lipschitz.pkl".format(results_dir), "wb") as f:
        pickle.dump(data, f)
                
if __name__ == "__main__":
    for exp_num in range(1, 169):
        global_lipschitz(exp_num)
        print("##############################################")

            