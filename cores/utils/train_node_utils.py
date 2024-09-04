import time
import torch
import os
import numpy as np
from .utils import format_time, get_grad_l2_norm
from cores.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from cores.utils.utils import save_nn_weights


def train_node(model, full_dynamics, train_config, trainloader, train_transform, nn_config, device, criterion, results_dir):
    # model should already be embedded in full_dynamics
    if train_config["adjoint"]:
        from cores.torchdiffeq import odeint_adjoint as odeint
    else:
        from cores.torchdiffeq import odeint

    model.train()
    num_epoch = train_config["num_epoch"]
    if nn_config.layer == 'Plain':
        if train_transform:
            transform_params = [p for name, p in model.named_parameters() 
                                if ('input_transform' in name) or ('output_transform' in name)]
            other_params = [p for name, p in model.named_parameters() 
                            if ('input_transform' not in name) and ('output_transform' not in name)]
            optimizer = torch.optim.Adam([
                            {'params': transform_params, 'weight_decay': train_config['transform_wd']},
                            {'params': other_params, 'weight_decay': train_config['wd']}],
                            lr=train_config['lr'])
        else: 
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    elif nn_config.layer == 'Sandwich': 
        if train_transform:
            transform_params = [p for name, p in model.named_parameters() 
                                if ('input_transform' in name) or ('output_transform' in name)]
            other_params = [p for name, p in model.named_parameters() 
                            if ('input_transform' not in name) and ('output_transform' not in name)]
            optimizer = torch.optim.Adam([
                            {'params': transform_params, 'weight_decay': train_config['transform_wd']},
                            {'params': other_params, 'weight_decay': 0.0}],
                            lr=train_config['lr'])
        else: 
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    else:
        raise ValueError(f"Unsupported layer: {nn_config.layer}")
    scheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=train_config['lr'], min_lr=0.0, 
                                              first_cycle_steps=num_epoch, warmup_steps=train_config["warmup_steps"])
    print("==> Number of param_groups in optimizer:", len(optimizer.param_groups))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    clip_value = train_config["clip_value"]
    loss_monitor = []
    grad_norm_monitor = []
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epoch):
        epoch_loss = 0
        epoch_grad_norm = 0
        epoch_start_time = time.time()
        for batch_idx, (t, x0, x) in enumerate(trainloader):
            t = t[0].squeeze()
            t = t.to(device)
            x0 = x0.to(device)
            x = x.reshape(-1,x.shape[2]).to(device)
            if train_config["adjoint"]:
                pred_x = odeint(full_dynamics, x0, t, adjoint_params=trainable_params).to(device)
            else:
                pred_x = odeint(full_dynamics, x0, t).to(device)
            pred_x = pred_x.permute(1,0,2)
            pred_x = pred_x.reshape(-1,pred_x.shape[2])
            optimizer.zero_grad()
            loss = criterion(x, pred_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_value)
            grad_norm = get_grad_l2_norm(model)
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.detach().cpu().numpy()
                epoch_grad_norm += grad_norm
        epoch_end_time = time.time()
        scheduler.step()
        if (epoch+1)%1 == 0:
            print("Epoch: {:03d} | Loss: {:.7f} | Grad norm: {:.6f} | Time: {}".format(epoch+1,
            epoch_loss/(batch_idx+1), epoch_grad_norm/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        loss_monitor.append(epoch_loss/(batch_idx+1))
        grad_norm_monitor.append(epoch_grad_norm/(batch_idx+1))
        if loss_monitor[-1] < best_loss:
            best_loss = loss_monitor[-1]
            print("==> Saving best model weights...")
            save_nn_weights(model, full_path=os.path.join(results_dir, 'nn_best.pt'))
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))

    return loss_monitor, grad_norm_monitor

    