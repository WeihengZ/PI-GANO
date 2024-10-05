import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable

from .utils_losses import darcy_loss

# plotting function
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=5)
    # Add a colorbar
    plt.colorbar(label='f')

# validation function
def val(model, loader, device, args, num_nodes_list):

    # extract the information of node numbers
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    mean_relative_L2 = 0
    num_eval = 0
    for (par, coors, u, flag, par_flag) in loader:

        par = par.float().to(device)
        par_flag = par_flag.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)

        # extract shape coordinates
        if args.geo_node == 'vary_bound' or 'vary_bound_sup': 
            ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)
        shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

        # model forward
        pred = model(coors[:,:,0], coors[:,:,1], par, par_flag, shape_coor, shape_flag)

        # get the flag
        flag_valid = torch.where(flag>=0, torch.ones_like(flag), torch.zeros_like(flag)).float().to(device)

        L2_relative = (torch.norm(pred*flag_valid-u*flag_valid, dim=-1) / torch.norm(u*flag_valid, dim=-1))
        mean_relative_L2 += torch.sum(L2_relative)
        num_eval += par.shape[0]

    mean_relative_L2 /= num_eval
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    return mean_relative_L2

# testing function
def test(model, loader, device, args, num_nodes_list):

    # extract the information of node numbers
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    mean_relative_L2 = 0
    num_eval = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (par, coors, u, flag, par_flag) in loader:
        par = par.float().to(device)
        par_flag = par_flag.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        flag = flag.float().to(device)

        # extract shape coordinates
        if args.geo_node == 'vary_bound' or 'vary_bound_sup': 
            ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)
        shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

        # model forward
        pred = model(coors[:,:,0], coors[:,:,1], par, par_flag, shape_coor, shape_flag)

        # get the flag
        flag_valid = torch.where(flag>=0, torch.ones_like(flag), torch.zeros_like(flag)).float().to(device)

        L2_relative = (torch.norm(pred*flag_valid-u*flag_valid, dim=-1) / torch.norm(u*flag_valid, dim=-1))

        # find the max and min error sample in this batch
        max_err, max_err_idx = torch.topk(L2_relative, 1)
        if max_err > max_relative_err:
            max_relative_err = max_err
            worst_xcoor = coors[max_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_ycoor = coors[max_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_f = pred[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            worst_gt = u[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            worst_ff = flag[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(worst_ff>=-0.1)[0]
            worst_xcoor = worst_xcoor[valid_id]
            worst_ycoor = worst_ycoor[valid_id]
            worst_f = worst_f[valid_id]
            worst_gt = worst_gt[valid_id]
            worst_ff = worst_ff[valid_id]
        min_err, min_err_idx = torch.topk(-L2_relative, 1)
        min_err = -min_err
        if min_err < min_relative_err:
            min_relative_err = min_err
            best_xcoor = coors[min_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            best_ycoor = coors[min_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            best_f = pred[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            best_gt = u[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            best_ff = flag[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(best_ff>=-0.1)[0]
            best_xcoor = best_xcoor[valid_id]
            best_ycoor = best_ycoor[valid_id]
            best_f = best_f[valid_id]
            best_gt = best_gt[valid_id]
            best_ff = best_ff[valid_id]
            

        # compute average error
        mean_relative_L2 += torch.sum(L2_relative)
        num_eval += par.shape[0]

    mean_relative_L2 /= num_eval
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    # color bar range
    max_color = np.amax([np.amax(worst_gt), np.amax(best_gt)])
    min_color = np.amin([np.amin(worst_gt), np.amin(best_gt)])

    # make the plot
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15,8))
    plt.subplot(2,3,1)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2,3,2)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.scatter(worst_xcoor, worst_ycoor, c=np.abs(worst_f-worst_gt), cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('absolute error')
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.scatter(best_xcoor, best_ycoor, c=best_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2,3,5)
    plt.scatter(best_xcoor, best_ycoor, c=best_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.scatter(best_xcoor, best_ycoor, c=np.abs(best_f-best_gt), cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('absolute error')
    plt.colorbar()
    plt.savefig(r'./res/plots/sample_{}_{}_{}.png'.format(args.geo_node, args.model, args.data))

    return mean_relative_L2

# define the training function
def train(args, config, model, device, loaders, num_nodes_list):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []

    # move the model to the defined device
    try:
        model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model), map_location=device))  
    except:
        print('No trained models') 
    model = model.to(device)

    # define tradeoff weights
    weight_bc = config['train']['weight_bc']
    weight_pde = config['train']['weight_pde']

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_bc_loss = np.inf
        for e in pbar:
          
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, device, args, num_nodes_list)
                print('Current epoch error:', err)
                print('current epochs pde loss:', avg_pde_loss, 'bc loss:', avg_bc_loss)

                avg_pde_loss = 0
                avg_bc_loss = 0
                if err < min_val_err:
                    torch.save(model.state_dict(), r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            for (par, coors, u, flag, par_flag) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # random sampling for PDE residual computation
                    ss_index = np.random.choice(np.arange(max_pde_nodes), config['train']['coor_sampling_size'])
                    pde_sampled_coors = coors[:, ss_index, :]
                    pde_sampled_coors = pde_sampled_coors.float().to(device)
                    pde_flag = flag[:, ss_index]
                    pde_flag = torch.where(pde_flag>-0.5, torch.ones_like(pde_flag), torch.zeros_like(pde_flag))
                    pde_flag = pde_flag.float().to(device)

                    # extract bc condition coordinates
                    ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
                    bc_coors = coors[:, ss_index, :].float().to(device)
                    bc_flag = flag[:, ss_index]
                    bc_flag = torch.where(bc_flag>0, torch.ones_like(bc_flag), torch.zeros_like(bc_flag))
                    bc_flag = bc_flag.float().to(device)
                    u_BC_gt = u[:, ss_index].float().to(device)

                    # prepare the parameter input
                    par = par.float().to(device)
                    par_flag = par_flag.float().to(device)

                    # prepare the shape coordinate input
                    if args.geo_node == 'vary_bound' or 'vary_bound_sup': 
                        ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)
                    shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
                    shape_flag = flag[:, ss_index]
                    shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

                    # forward to get the prediction on fixed boundary
                    u_BC_pred = model(bc_coors[:,:,0], bc_coors[:,:,1], par, par_flag, shape_coor, shape_flag)
                    
                    # forward to get the prediction on pde domian
                    x_pde = Variable(pde_sampled_coors[:,:,0], requires_grad=True)
                    y_pde = Variable(pde_sampled_coors[:,:,1], requires_grad=True)
                    u_pde_pred = model(x_pde, y_pde, par, par_flag, shape_coor, shape_flag)

                    # compute the losses
                    pde_loss = darcy_loss(u_pde_pred, x_pde, y_pde, pde_flag)
                    bc_loss = mse(u_BC_pred*bc_flag, u_BC_gt*bc_flag)
                    total_loss = weight_pde*pde_loss + weight_bc*bc_loss

                    # store the loss
                    avg_pde_loss += pde_loss.detach().cpu().item()
                    avg_bc_loss += bc_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

    # final test
    model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model), map_location=device))   
    model.eval()
    err = test(model, test_loader, device, args, num_nodes_list)
    print('Best L2 relative error on test loader:', err)

# define the supervised training function
def sup_train(args, config, model, device, loaders, num_nodes_list):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []

    # move the model to the defined device
    try:
        model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model), map_location=device))  
    except:
        print('No trained models') 
    model = model.to(device)

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_bc_loss = np.inf
        for e in pbar:
          
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, device, args, num_nodes_list)
                print('Current epoch error:', err)
                print('current epochs pde loss:', avg_pde_loss, 'bc loss:', avg_bc_loss)

                avg_pde_loss = 0
                avg_bc_loss = 0
                if err < min_val_err:
                    torch.save(model.state_dict(), r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            for (par, coors, u, flag, par_flag) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # extract bc condition coordinates
                    all_coors = coors[:, :, :].float().to(device)
                    all_flag = flag[:, :]
                    all_flag = torch.where(all_flag>-0.5, torch.ones_like(all_flag), torch.zeros_like(all_flag)).float().to(device)
                    gt = u[:, :].float().to(device)

                    # prepare the parameter input
                    par = par.float().to(device)
                    par_flag = par_flag.float().to(device)

                    # prepare the shape coordinate input
                    if args.geo_node == 'vary_bound_sup':
                        ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)
                    shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
                    shape_flag = flag[:, ss_index]
                    shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

                    # forward to get the prediction on fixed boundary
                    u_pred = model(all_coors[:,:,0], all_coors[:,:,1], par, par_flag, shape_coor, shape_flag)

                    # compute the losses
                    total_loss = mse(u_pred*all_flag, gt*all_flag)

                    # store the loss
                    avg_pde_loss += total_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

    # final test
    model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model), map_location=device))   
    model.eval()
    err = test(model, test_loader, device, args, num_nodes_list)
    print('Best L2 relative error on test loader:', err)