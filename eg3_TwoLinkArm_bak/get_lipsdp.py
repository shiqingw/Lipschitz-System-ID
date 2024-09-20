import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scipy.io import savemat
import matlab.engine
from time import time
import pickle 

from cores.lip_nn.models import NeuralNetwork
from cores.utils.config import Configuration
from cores.utils.utils import get_nn_config, load_nn_weights

def convert(exp_num):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
        print("==> convert() does not support Sandwich layer. Skipping exp_num:", exp_num)
        return

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()

    weights = []
    net_dims = []

    # input transform
    input_transform = model.input_transform.cpu().detach().numpy()
    input_transform = np.diag(input_transform)
    weights.append(input_transform.astype(np.float64))
    net_dims.append(input_transform.shape[0])
    net_dims.append(input_transform.shape[1])

    # hidden layers
    for layer in model.model:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.cpu().detach().numpy().astype(np.float64))
            net_dims.append(layer.weight.cpu().detach().numpy().shape[0])
    
    # output transform
    output_transform = model.output_transform.cpu().detach().numpy()
    output_transform = np.diag(output_transform)
    weights.append(output_transform.astype(np.float64))
    net_dims.append(output_transform.shape[0])

    # sanity check
    assert len(net_dims) == len(weights) + 1
    for i in range(1, len(net_dims)):
        if net_dims[i] != weights[i-1].shape[0]:
            raise ValueError("Dimension mismatch.")
    
    # save to file
    print("==> Saving to file...")
    file_name = os.path.join(results_dir, "lipsdp_weights.mat")
    data = {'weights': np.array(weights, dtype=object)}
    savemat(file_name, data)

    print("==> Saved to:", file_name)

def run_lipsdp(exp_num):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    nn_config = test_settings["nn_config"]
    if nn_config["layer"] == 'Sandwich':
        print("==> run_lipsdp() does not support Sandwich layer. Skipping exp_num:", exp_num)
        return

    # run lipsdp
    start_time = time()
    eng = matlab.engine.start_matlab()
    eng.addpath(r'LipSDP/matlab_engine')
    eng.addpath(r'LipSDP/matlab_engine/weight_utils')
    eng.addpath(r'LipSDP/matlab_engine/error_messages')

    if nn_config["activations"] == 'leaky_relu':
        alpha = 0.01
        beta = 1.0
    else:
        alpha = 0.0
        beta = 1.0
    network = {
        'alpha': alpha,
        'beta': beta,
        'weight_path': [str(os.path.join(results_dir, "lipsdp_weights.mat"))],
    }

    lip_params = {
        'formulation': 'neuron',
        'split': matlab.logical([False]),
        'parallel': matlab.logical([False]),
        'verbose': matlab.logical([False]),
        'split_size': matlab.double([0]),
        'num_neurons': matlab.double([0]),
        'num_workers': matlab.double([0]),
        'num_dec_vars': matlab.double([0])
    }

    L = eng.solve_LipSDP(network, lip_params, nargout=1)
    print(f'LipSDP computes a Lipschitz constant of {L:.3f}')
    print(f'Total time: {float(time() - start_time):.5} seconds')
    print("==> Saving to file...")

    # save to file
    file_name = os.path.join(results_dir, "lipsdp_lipschitz.pkl")
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        data['lipsdp_lipschitz'] = L
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    else:
        data = {'lipsdp_lipschitz': L}
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    print("==> Done saving lipsdp_lipschitz")
    

if __name__ == '__main__':
    exp_nums = [269, 270, 271, 272]
    for exp_num in exp_nums:
        convert(exp_num)
        run_lipsdp(exp_num)
        print("##############################################")
    
