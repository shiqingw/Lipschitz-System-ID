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
    state_space = np.array([[-np.pi/4, np.pi/4],
                            [-np.pi/4, np.pi/4],
                            [-0.01, 0.01],
                            [-0.01, 0.01]], dtype=config.np_dtype)

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

    
if __name__ == "__main__":

    exp_nums = [81, 82, 83, 84, 161, 162, 163, 164, 261, 262, 263, 264]
    for exp_num in exp_nums:
        diagnosis(exp_num)
        print("#############################################")