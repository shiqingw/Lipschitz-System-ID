import json
import sys
import os
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torchinfo import summary
from torch.utils.data import DataLoader

from cores.lip_nn.models import NeuralNetwork
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, get_nn_config, save_nn_weights, save_dict
from cores.utils.config import Configuration
from cores.utils.train_utils import train_nn, train_lip_regularized
from cores.dataloader.dataset_utils import DynDataset, get_test_and_training_data
from cores.utils.draw_utils import draw_curve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=193, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    # Load test settings
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

    # Load dataset and get training set and testing set trainloader
    train_config = test_settings["train_config"]
    dataset_num = int(train_config["dataset"])
    dataset_path = "eg3_TwoLinkArm/{:03d}/dataset.mat".format(dataset_num)
    dataset_path = "{}/datasets/{}".format(str(Path(__file__).parent.parent),dataset_path)
    dataset = DynDataset(dataset_path, config)

    train_ratio = train_config["train_ratio"]
    further_train_ratio = train_config["further_train_ratio"]
    seed_train_test = train_config["seed_train_test"]
    seed_actual_train = train_config["seed_actual_train"]
    train_dataset, test_dataset = get_test_and_training_data(dataset, train_ratio, further_train_ratio, seed_train_test=None, seed_actual_train=None)

    batch_size = train_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epoch = train_config["num_epoch"]

    # Build dynamical system
    system_name = test_settings["nominal_system_name"]
    system = get_system(system_name).to(device)

    # Build neural network
    nn_config = test_settings["nn_config"]
    input_bias = np.array(nn_config["input_bias"], dtype=config.np_dtype)
    input_transform = np.array(nn_config["input_transform_to_inverse"], dtype=config.np_dtype)
    input_transform = 1.0/input_transform
    output_transform = np.array(nn_config["output_transform"], dtype=config.np_dtype)
    train_transform = bool(nn_config["train_transform"])
    zero_at_zero = bool(nn_config["zero_at_zero"])

    nn_config = get_nn_config(nn_config)
    if nn_config.layer == 'Lip_Reg':
        model = NeuralNetwork(nn_config, input_bias=None, input_transform=None, output_transform=None, train_transform=False, zero_at_zero=False)
    else:
        model = NeuralNetwork(nn_config, input_bias, input_transform, output_transform, train_transform, zero_at_zero)
    if nn_config.layer == 'Sandwich':
        print("==> Lipschitz constant: {:.02f}".format(nn_config.gamma))

    print("==> Input bias to be applied to the neural network:")
    print(model.input_bias.detach().cpu().numpy())
    print("==> Input transform to be applied to the neural network:")
    print(model.input_transform.detach().cpu().numpy())
    print("==> Ouput transform to be applied to the neural network:")
    print(model.output_transform.detach().cpu().numpy())
    print('==> Evaluating model...')
    summary(model, input_size=(1, system.n_state))
    print("==> Saving initial model weights...")
    model = model.to(device)
    save_nn_weights(model, full_path=os.path.join(results_dir, 'nn_init.pt'))

    # Define criterion 
    criterion = lambda x, y: torch.nn.functional.pairwise_distance(x, y).square().mean()
    residual_func = lambda x, u: model(x)

    # Training
    print("==> Start training...")
    selected_indices = [2,3]
    best_loc = os.path.join(results_dir, 'nn_best.pt')
    if nn_config.layer == "Lip_Reg":
        train_loss_monitor, train_l2_loss_monitor, train_lip_loss_monitor, train_grad_norm_monitor, test_loss_monitor = \
            train_lip_regularized(model, residual_func, system, criterion, train_config, device, train_loader, test_loader, selected_indices, best_loc)
    elif nn_config.layer == 'Plain' or nn_config.layer == 'Sandwich':
        train_loss_monitor, train_grad_norm_monitor, test_loss_monitor = \
            train_nn(model, residual_func, system, criterion, train_config, train_transform, device, train_loader, test_loader, selected_indices, best_loc)
    else:
        raise ValueError(f"Unsupported layer: {nn_config.layer}")
    
    print("==> Saving trained model weights...")
    save_nn_weights(model, full_path=os.path.join(results_dir, 'nn_trained.pt'))

    print("==> Input bias to be applied to the neural network (trained):")
    print(model.input_bias.cpu().detach().numpy())
    print("==> Input transform to be applied to the neural network (trained):")
    print(model.input_transform.cpu().detach().numpy())
    print("==> Output transform to be applied to the neural network (trained):")
    print(model.output_transform.cpu().detach().numpy())

    print("==> Saving training info...")
    training_info = {"train_loss": train_loss_monitor,
                     "train_grad_norm": train_grad_norm_monitor,
                     "test_loss": test_loss_monitor}

    if nn_config.layer == "Lip_Reg":
        training_info["train_l2_loss"] = train_l2_loss_monitor
        training_info["train_lip_loss"] = train_lip_loss_monitor
    save_dict(training_info, os.path.join(results_dir, "training_info.npy"))

    # Draw training loss curve
    print("==> Drawing training loss...")
    draw_curve(train_loss_monitor, num_epoch, config, ylabel="loss", results_dir=os.path.join(results_dir, 'train_loss.pdf'))

    # Draw training grad norm curve
    print("==> Drawing grad norm...")
    draw_curve(train_grad_norm_monitor, num_epoch, config, ylabel="grad norm", results_dir=os.path.join(results_dir, 'train_grad_norm.pdf'))

    # Draw testing loss curve
    print("==> Drawing testing loss...")
    draw_curve(test_loss_monitor, num_epoch, config, ylabel="loss", results_dir=os.path.join(results_dir, 'test_loss.pdf'))

    if nn_config.layer == "Lip_Reg":
        # Draw training loss curve
        print("==> Drawing l2 loss...")
        draw_curve(train_l2_loss_monitor, num_epoch, config, ylabel="l2 loss", results_dir=os.path.join(results_dir, 'l2_loss.pdf'))

        # Draw training loss curve
        print("==> Drawing lip loss...")
        draw_curve(train_lip_loss_monitor, num_epoch, config, ylabel="lip loss", results_dir=os.path.join(results_dir, 'lip_loss.pdf'))

    print("==> Process finished.")