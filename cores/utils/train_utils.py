import time
import torch
import random
from .utils import format_time, get_grad_l2_norm
from cores.cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def train_nn(model, residual_func, system, criterion, train_config, train_transform, device, train_loader, test_loader, selected_indices, best_loc):
    # Define optimizer, learning rate scheduler, loss function, and loss monitor
    num_epoch = train_config["num_epoch"]
    if model.layer == 'Plain':
        if train_transform:
            transform_params = [p for name, p in model.named_parameters() 
                                if ('input_transform' in name) or ('output_transform' in name)]
            other_params = [p for name, p in model.named_parameters() 
                            if ('input_transform' not in name) and ('output_transform' not in name)]
            optimizer = torch.optim.Adam([
                            {'params': transform_params, 'weight_decay': train_config['transform_wd'], 'lr': train_config['transform_lr']},
                            {'params': other_params, 'weight_decay': train_config['wd'], 'lr': train_config['lr']}])
        else: 
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    elif model.layer == 'Sandwich': 
        if train_transform:
            transform_params = [p for name, p in model.named_parameters() 
                                if ('input_transform' in name) or ('output_transform' in name)]
            other_params = [p for name, p in model.named_parameters() 
                            if ('input_transform' not in name) and ('output_transform' not in name)]
            optimizer = torch.optim.Adam([
                            {'params': transform_params, 'weight_decay': train_config['transform_wd'], 'lr': train_config['transform_lr']},
                            {'params': other_params, 'weight_decay': 0.0, 'lr': train_config['lr']}])
        else: 
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    else:
        raise ValueError(f"Unsupported layer: {model.layer}")
    scheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=train_config['lr'], min_lr=0.0, 
                                              first_cycle_steps=num_epoch, warmup_steps=train_config["warmup_steps"])
    print("==> Number of param_groups in optimizer:", len(optimizer.param_groups))

    train_loss_monitor = []
    test_loss_monitor = []
    train_grad_norm_monitor = []
    best_epoch_test_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epoch):
        # Train
        model.train()
        epoch_train_loss = 0
        epoch_train_grad_norm = 0
        epoch_train_start_time = time.time()
        for batch_idx, (_, x, u, x_dot) in enumerate(train_loader):
            x = x.to(device)
            u = u.to(device)
            x_dot = x_dot.to(device)
            optimizer.zero_grad()
            residual = residual_func(x,u)
            nominal = system(x,u)
            loss = criterion(x_dot[:,selected_indices], residual + nominal[:,selected_indices])
            loss.backward()
            grad_norm = get_grad_l2_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_train_loss += loss.detach().cpu().numpy()
                epoch_train_grad_norm += grad_norm
        epoch_train_end_time = time.time()
        epoch_train_loss = epoch_train_loss/(batch_idx+1)
        epoch_train_grad_norm = epoch_train_grad_norm/(batch_idx+1)
        print("Epoch: {:03d} | Train Loss: {:.7f} | Grad norm: {:.6f} | Time: {}".format(epoch+1,
                        epoch_train_loss, epoch_train_grad_norm, format_time(epoch_train_end_time - epoch_train_start_time)))
        train_loss_monitor.append(epoch_train_loss)
        train_grad_norm_monitor.append(epoch_train_grad_norm)
        
        # Test
        model.eval()
        epoch_test_loss = 0
        epoch_test_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (_, x, u, x_dot) in enumerate(test_loader):
                x = x.to(device)
                u = u.to(device)
                x_dot = x_dot.to(device)
                residual = residual_func(x,u)
                nominal = system(x,u)
                loss = criterion(x_dot[:,selected_indices], residual + nominal[:,selected_indices])
                epoch_test_loss += loss.detach().cpu().numpy()
        epoch_test_end_time = time.time()
        epoch_test_loss = epoch_test_loss/(batch_idx+1)
        print("Epoch: {:03d} | Test Loss: {:.7f} | Time: {}".format(epoch+1,
                        epoch_test_loss, format_time(epoch_test_end_time - epoch_test_start_time)))
        test_loss_monitor.append(epoch_test_loss)

        # Save the model if the test loss is the best
        if epoch_test_loss < best_epoch_test_loss:
            best_epoch_test_loss = epoch_test_loss
            torch.save(model.state_dict(), best_loc)
            print("==> Save the model at epoch {:03d} with test loss {:.7f}".format(epoch+1, best_epoch_test_loss))

        # Step the learning rate scheduler
        scheduler.step()
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    return train_loss_monitor, train_grad_norm_monitor, test_loss_monitor

def Quot(model, input1, input2):
    output1 = model(input1)
    output2 = model(input2)
    Num = torch.norm(output1-output2, 2)
    Den = torch.norm(input1-input2, 2)
    div = torch.div(Num,Den)
    return div

#select how_many_points random points in the input and compute and approximate Lipschitz constant with them.
def Lip(model, input_matrix, device):
    how_many_points = 10
    which_points = random.sample(range(0, len(input_matrix)), how_many_points)    
    Quot_vec = torch.empty(0, 1, device=device)
    for in1 in range(len(which_points)):
        for in2 in range(in1+1, len(which_points)-1):
            first_pt = input_matrix[which_points[in1],:].reshape(1,-1)
            second_pt = input_matrix[which_points[in2],:].reshape(1,-1)
            Quot_vec = torch.cat((Quot_vec, Quot(model, first_pt, second_pt).reshape(-1,1)))
    maxim = torch.max(Quot_vec)
    return maxim


def train_lip_regularized(model, residual_func, system, criterion, train_config, device, train_loader, test_loader, selected_indices, best_loc):
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False) 
    reg_param = train_config['lip_reg_param']
    train_lip_loss_monitor = []
    train_l2_loss_monitor = []
    train_loss_monitor = []
    train_grad_norm_monitor = []
    test_loss_monitor = []
    best_epoch_test_loss = float('inf')
    num_epoch = train_config["num_epoch"]

    start_time = time.time()
    for epoch in range(num_epoch):
        # Train
        model.train()
        epoch_train_loss = 0
        epoch_train_grad_norm = 0
        epoch_train_l2_loss = 0
        epoch_train_lip_loss = 0
        epoch_train_start_time = time.time()

        if epoch % 5 == 0: #decrease learning rate every 5 epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.

        for batch_idx, (_, x, u, x_dot) in enumerate(train_loader):
            x = x.to(device)
            u = u.to(device)
            x_dot = x_dot.to(device)
            optimizer.zero_grad()
            residual = residual_func(x,u)
            nominal = system(x,u)
            l2_loss = criterion(x_dot[:,selected_indices], residual + nominal[:,selected_indices])
            lip_loss = reg_param * Lip(model, x, device)
            loss = l2_loss + lip_loss
            loss.backward()
            grad_norm = get_grad_l2_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_train_loss += loss.detach().cpu().numpy()
                epoch_train_l2_loss += l2_loss.detach().cpu().numpy()
                epoch_train_lip_loss += lip_loss.detach().cpu().numpy()
                epoch_train_grad_norm += grad_norm

        epoch_train_end_time = time.time()
        epoch_train_loss = epoch_train_loss/(batch_idx+1)
        epoch_train_l2_loss = epoch_train_l2_loss/(batch_idx+1)
        epoch_train_lip_loss = epoch_train_lip_loss/(batch_idx+1)
        epoch_train_grad_norm = epoch_train_grad_norm/(batch_idx+1)

        print("Epoch: {:03d} | Loss: {:.7f} | L2 loss: {:.7f} | Lip loss: {:.7f} | Grad norm: {:.6f} | Time: {}".format(epoch+1,
            epoch_train_loss, epoch_train_l2_loss, epoch_train_lip_loss,
            epoch_train_grad_norm, format_time(epoch_train_end_time - epoch_train_start_time)))
        
        train_loss_monitor.append(epoch_train_loss)
        train_grad_norm_monitor.append(epoch_train_grad_norm)
        train_lip_loss_monitor.append(epoch_train_lip_loss)
        train_l2_loss_monitor.append(epoch_train_l2_loss)

        # Test
        model.eval()
        epoch_test_loss = 0
        epoch_test_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (_, x, u, x_dot) in enumerate(test_loader):
                x = x.to(device)
                u = u.to(device)
                x_dot = x_dot.to(device)
                residual = residual_func(x,u)
                nominal = system(x,u)
                loss = criterion(x_dot[:,selected_indices], residual + nominal[:,selected_indices])
                epoch_test_loss += loss.detach().cpu().numpy()
        epoch_test_end_time = time.time()
        epoch_test_loss = epoch_test_loss/(batch_idx+1)
        print("Epoch: {:03d} | Test Loss: {:.7f} | Time: {}".format(epoch+1,
                        epoch_test_loss, format_time(epoch_test_end_time - epoch_test_start_time)))
        test_loss_monitor.append(epoch_test_loss)

        # Save the model if the test loss is the best
        if epoch_test_loss < best_epoch_test_loss:
            best_epoch_test_loss = epoch_test_loss
            torch.save(model.state_dict(), best_loc)
            print("==> Save the model at epoch {:03d} with test loss {:.7f}".format(epoch+1, best_epoch_test_loss))

    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    return train_loss_monitor, train_l2_loss_monitor, train_lip_loss_monitor, train_grad_norm_monitor, test_loss_monitor