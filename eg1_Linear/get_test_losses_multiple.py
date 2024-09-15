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

def diagnosis(exp_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
    further_train_ratio = test_settings["train_config"]["further_train_ratio"]
    true_system = get_system(true_system_name).to(device)
    state_dim = nominal_system.n_state
    control_dim = nominal_system.n_control

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
        print("==> Lipschitz constant: {:.02f}".format(nn_config.gamma))

    model = load_nn_weights(model, os.path.join(results_dir, 'nn_best.pt'), device)
    model.eval()

    # Load loss dict and print the final loss
    loss_dict = load_dict(os.path.join(results_dir, "training_info.npy"))
    min_pos = np.argmin(loss_dict["test_loss"])
    
    print("==> Training loss: ", loss_dict["train_loss"][min_pos])
    print("==> Testing loss: ", loss_dict["test_loss"][min_pos])

    if nn_config.layer == 'Sandwich':
        overall_lipschitz = nn_config.gamma
        overall_lipschitz *= max(model.input_transform.cpu().detach().numpy())
        overall_lipschitz *= max(model.output_transform.cpu().detach().numpy())

    train_config = test_settings["train_config"]
    if nn_config.layer == 'Sandwich':
        return nn_config.gamma, overall_lipschitz, loss_dict["train_loss"][min_pos], loss_dict["test_loss"][min_pos], exp_num, further_train_ratio
    elif nn_config.layer == 'Plain':
        return train_config['wd'], loss_dict["train_loss"][min_pos], loss_dict["test_loss"][min_pos], exp_num, further_train_ratio
    elif nn_config.layer == 'Lip_Reg':
        return train_config['lip_reg_param'], loss_dict["train_loss"][min_pos], loss_dict["test_loss"][min_pos], exp_num, further_train_ratio

if __name__ == "__main__":
    # save to a txt file with separator that can be directly copy pasted to excel-
    with open("text.txt", "w") as file:
        exp_nums = list(range(1, 169))
        for exp_num in exp_nums:
            out = diagnosis(exp_num)
            print("#############################################")
            for ii in out:
                file.write(f"{ii}\t")
            file.write(f"\n")