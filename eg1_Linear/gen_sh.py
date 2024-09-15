import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script(filename, exp_nums, device):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg1_results/{exp_num:03}\n"
            command2 = f"python -u eg1_Linear/train.py --exp_num {exp_num} --device {device} > eg1_results/{exp_num:03}/output.out\n"
            file.write(command1)
            file.write(command2)

exp_nums = list(range(1, 265))
devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
files = [os.path.join(str(Path(__file__).parent.parent), f"run_cuda_{i}.sh") for i in range(len(devices))]

my_dic = {}
for d in devices:
    my_dic[d] = []

counter = 0
for exp_num in exp_nums:
    my_dic[devices[counter % len(devices)]].append(exp_num)
    counter = counter + 1

for i in range(len(devices)):
    file = files[i]
    device = devices[i]
    generate_sh_script(file, my_dic[device], device)

