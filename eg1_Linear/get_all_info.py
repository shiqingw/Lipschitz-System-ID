import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import pickle 

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, load_dict, load_nn_weights
from cores.utils.config import Configuration


def collect_info(exp_num, grid_sizes):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = os.path.join(results_dir, "test_settings_{:03d}.json".format(exp_num))
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    all_info = []

    # Step 1: Gamma/wd/lip_reg_param
    nn_config = test_settings["nn_config"]
    train_config = test_settings["train_config"]
    if nn_config["layer"] == "Sandwich":
        all_info.append(nn_config["Lipschitz_constant"])
    elif nn_config["layer"] == "Plain":
        all_info.append(train_config["wd"])
    elif nn_config["layer"] == "Lip_Reg":
        all_info.append(train_config["lip_reg_param"])
    
    # Step 2: Load loss dict and print the final loss
    loss_dict = load_dict(os.path.join(results_dir, "training_info.npy"))
    min_pos = np.argmin(loss_dict["test_loss"])
    train_loss_out = loss_dict["train_loss"][min_pos]
    test_loss_out = loss_dict["test_loss"][min_pos]
    all_info.append(train_loss_out)
    all_info.append(test_loss_out)

    # Step 4: exp_num
    all_info.append(exp_num)

    # Step 5: further_train_ratio
    further_train_ratio = train_config["further_train_ratio"]
    all_info.append(further_train_ratio)

    # Step 6: Lipschitz constant, both global_lipschitz and lipsdp_lipschitz
    global_lipschitz_file = os.path.join(results_dir, "global_lipschitz.pkl")
    if os.path.exists(global_lipschitz_file):
        with open(global_lipschitz_file, "rb") as f:
            global_lipschitz_data = pickle.load(f)
        all_info.append(global_lipschitz_data["global_lipschitz"])
    else:
        all_info.append(None)

    lipsdp_lipschitz_file = os.path.join(results_dir, "lipsdp_lipschitz.pkl")
    if os.path.exists(lipsdp_lipschitz_file):
        with open(lipsdp_lipschitz_file, "rb") as f:
            lipsdp_lipschitz_data = pickle.load(f)
        all_info.append(lipsdp_lipschitz_data["lipsdp_lipschitz"])
    else:
        all_info.append(None)

    # Step 7: estimation_error, both global_lipschitz_error and lipsdp_lipschitz_error
    for grid_size in grid_sizes:
        estimation_error_file = os.path.join(results_dir, "estimate_error_grid_size_{:.2f}.pkl".format(grid_size))
        if os.path.exists(estimation_error_file):
            with open(estimation_error_file, "rb") as f:
                estimation_error_data = pickle.load(f)
                # check if has the key
                if "global_lipschitz_error" in estimation_error_data:
                    all_info.append(estimation_error_data["global_lipschitz_error"])
                else:
                    all_info.append(None)
                if "lipsdp_lipschitz_error" in estimation_error_data:
                    all_info.append(estimation_error_data["lipsdp_lipschitz_error"])
                else:    
                    all_info.append(None)
        else:
            all_info.append(None)
            all_info.append(None)

    # Step 8: Time for computing
    for grid_size in grid_sizes:
        estimation_error_file = os.path.join(results_dir, "estimate_error_grid_size_{:.2f}.pkl".format(grid_size))
        if os.path.exists(estimation_error_file):
            with open(estimation_error_file, "rb") as f:
                estimation_error_data = pickle.load(f)
                # check if has the key
                if "global_lipschitz_estimation_time" in estimation_error_data:
                    all_info.append(estimation_error_data["global_lipschitz_estimation_time"])
                else:
                    all_info.append(None)
                if "lipsdp_lipschitz_estimation_time" in estimation_error_data:
                    all_info.append(estimation_error_data["lipsdp_lipschitz_estimation_time"])
                else:    
                    all_info.append(None)
        else:
            all_info.append(None)
            all_info.append(None)

    
    return all_info

if __name__ == "__main__":
    # Load the entire Excel file
    excel_file = '/Users/shiqing/Desktop/Lipschitz-System-ID/eg1.xlsx'
    # Load all sheets
    xls = pd.ExcelFile(excel_file)
    # List sheet names
    all_sheet_names = xls.sheet_names
    exp_nums = []
    for sheet_name in all_sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        column_data = df['Exp Num'].tolist()
        exp_nums.extend(column_data)
    
    grid_sizes = [0.1, 0.05]

    # save to a txt file with separator that can be directly copy pasted to excel
    with open("text.txt", "w") as file:
        for exp_num in exp_nums:
            out = collect_info(exp_num, grid_sizes)
            for ii in out:
                file.write(f"{ii}\t")
            file.write(f"\n")
            print("#############################################")