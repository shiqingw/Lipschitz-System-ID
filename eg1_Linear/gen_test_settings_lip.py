import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_json_script(filename, entry):
    data = {
            "nominal_system_name": "LinearSystem1_nominal",
            "true_system_name": "LinearSystem1_true",
            "seed": 0,
            "nn_config": {
                "in_features": 2,
                "out_features": 2,
                "Lipschitz_constant": entry,
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
                "further_train_ratio": 1.0,
                "seed_train_test": 1,
                "seed_actual_train": 2,
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

data = [1, 2, 4, 8]
data.sort()
start = 9
exp_nums = range(start, start+len(data))
files = [os.path.join(str(Path(__file__).parent.parent), 'eg1_Linear', 'test_settings', f"test_settings_{exp_num:03}.json") for exp_num in exp_nums]

counter = 0
for exp_num in exp_nums:
    file = files[counter]
    entry = data[counter]
    generate_json_script(file, entry)
    counter = counter + 1