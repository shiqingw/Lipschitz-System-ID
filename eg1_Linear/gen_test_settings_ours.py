import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_json_script(filename, entry):
    data = {
            "nominal_system_name": "LinearSystem1_nominal",
            "true_system_name": "LinearSystem1_true",
            "seed": entry["random_seed"],
            "nn_config": {
                "in_features": 2,
                "out_features": 2,
                "Lipschitz_constant": entry["gamma"],
                "layer": "Sandwich",
                "activations": "relu",
                "num_layers": 8,
                "width_each_layer": 64,
                "input_bias": [
                    0.0,
                    0.0
                ],
                "input_transform_to_inverse": [
                    1.9646,
                    1.9646
                ],
                "output_transform": [
                    3.9366,
                    3.9363
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
                "wd": 0,
                "transform_wd": 0.0,
                "transform_lr": 0.0,
                "lip_reg_param": 0.0
            }
        }
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

train_ratios = [0.25, 0.5, 1.0]
train_ratios.sort()
gammas = [0.5, 1, 2, 4, 8, 16]
gammas.sort()
random_seeds = [0]
random_seeds.sort()

# iterate over all combinations of gammas, train_ratios, and random_seeds and generate a test_settings file for each
data = []
for train_ratio in train_ratios:
    for gamma in gammas:
        for random_seed in random_seeds:
            data.append({
                "gamma": gamma,
                "train_ratio": train_ratio,
                "random_seed": random_seed
            })

start = 1
exp_nums = range(start, start+len(data))
for i in range(len(data)):
    entry = data[i]
    exp_num = exp_nums[i]
    filename = os.path.join(str(Path(__file__).parent.parent), 'eg1_Linear', 'test_settings', f"test_settings_{exp_num:03}.json")
    generate_json_script(filename, entry)