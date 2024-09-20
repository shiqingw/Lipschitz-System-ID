import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_json_script(filename, entry):
    data = {
            "nominal_system_name": "TwoLinkArm2_nominal",
            "true_system_name": "TwoLinkArm2_true",
            "seed": entry["random_seed"],
            "nn_config": {
                "in_features": 4,
                "out_features": 2,
                "Lipschitz_constant": 0,
                "layer": "Plain",
                "activations": "relu",
                "num_layers": 8,
                "width_each_layer": 64,
                "input_bias": [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                "input_transform_to_inverse": [
                    1.5898, 
                    1.5901, 
                    0.0848, 
                    0.1039
                ],
                "output_transform": [
                    0.0466, 
                    0.0488
                ],
                "train_transform": 0,
                "zero_at_zero": 1
            },
            "train_config": {
                "dataset": "1",
                "train_ratio": 0.8,
                "further_train_ratio": entry["train_ratio"],
                "seed_train_test": "None",
                "seed_actual_train": "None",
                "batch_size": 256,
                "num_epoch": 40,
                "warmup_steps": 8,
                "lr": 0.001,
                "wd": entry["wd"],
                "transform_wd": 0.0,
                "transform_lr": 0.0,
                "lip_reg_param": 0.0
            }
        }
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

train_ratios = [0.25, 0.5, 1.0]

train_ratios.sort()
wds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
wds.sort()
random_seeds = [0,100,200,300]
random_seeds.sort()

# iterate over all combinations of gammas, train_ratios, and random_seeds and generate a test_settings file for each
data = []
for train_ratio in train_ratios:
    for wd in wds:
        for random_seed in random_seeds:
            data.append({
                "wd": wd,
                "train_ratio": train_ratio,
                "random_seed": random_seed
            })

start = 73
exp_nums = range(start, start+len(data))
for i in range(len(data)):
    entry = data[i]
    exp_num = exp_nums[i]
    filename = os.path.join(str(Path(__file__).parent.parent), 'eg3_TwoLinkArm', 'test_settings', f"test_settings_{exp_num:03}.json")
    generate_json_script(filename, entry)