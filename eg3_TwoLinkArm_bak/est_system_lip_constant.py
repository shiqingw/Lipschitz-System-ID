import os
import sys
import torch
import argparse
import numpy as np
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt

from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything
from cores.utils.config import Configuration
from cores.dataloader.dataset_utils import DynDataset

def diagnosis(exp_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=exp_num, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    exp_num = args.exp_num
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
    with open(os.path.join(str(Path(__file__).parent.parent), "cores", "dynamical_systems", "system_params.json"), 'r') as f:
        all_system = json.load(f)
        params = all_system[true_system_name]["params"]
    
    m_link1 = params["m_link1"]
    m_motor1 = params["m_motor1"]
    I_link1 = params["I_link1"]
    I_motor1 = params["I_motor1"]
    m_link2 = params["m_link2"]
    m_motor2 = params["m_motor2"]
    I_link2 = params["I_link2"]
    I_motor2 = params["I_motor2"]
    l1 = params["l1"]
    l2 = params["l2"]
    a1 = params["a1"]
    a2 = params["a2"]
    kr1 = params["kr1"]
    kr2 = params["kr2"]
    Fv1 = params["Fv1"] # viscous friction joint1 
    Fv2 = params["Fv2"] # viscous friction joint2 
    Fc1 = params["Fc1"] # Coulomb friction joint1
    Fc2 = params["Fc2"] # Coulomb friction joint2
    Fc_s1 = params["s1"]
    Fc_s2 = params["s2"]

    theta2 = np.linspace(-np.pi, np.pi, 100)
    min_M = 1e10
    for i in range(len(theta2)):
        cos2 = np.cos(theta2[i])
        M11 = I_link1 + m_link1 * l1**2 + kr1**2 * I_motor1 + I_link2 \
                + m_link2*(a1**2 + l2**2 + 2 * a1 * l2 * cos2) + I_motor2 \
                + m_motor2 * a1**2
        M12 = I_link2 + m_link2 * (l2**2 + a1 * l2 * cos2) + kr2 * I_motor2
        M21 = M12
        M22 = (I_link2 + m_link2 * l2**2 + kr2**2 * I_motor2)
        M = np.array([[M11, M12], [M21, M22]])
        eigvals = np.linalg.eigvals(M)
        min_M = min(min_M, min(eigvals))
    
    L = (max(Fv1, Fv2) + max(Fc1 * Fc_s1, Fc2 * Fc_s2))/min_M
    print("Lipschitz constant:", L)

if __name__ == "__main__":
    for exp_num in range(1, 2):
        diagnosis(exp_num)
        