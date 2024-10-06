import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from .utils_losses import plate_stress_loss, bc_edgeY_loss

# plotting function
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=5)
    # Add a colorbar
    plt.colorbar(label='f')

# validation function
def val(model, loader, args, device, num_nodes_list):

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes = num_nodes_list

    mean_relative_L2 = 0
    num_eval = 0
    for (par, coors, u, v, flag, par_flag) in loader:

        # extract domain shape information
        if args.geo_node == 'vary_bound' or 'vary_bound_sup':
            ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        if args.geo_node == 'all_bound':
            ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        shape_coors = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

        # prepare the data
        par = par.float().to(device)
        par_flag = par_flag.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)
        flag = flag.float().to(device)

        # model forward
        u_pred, v_pred = model(coors[:,:,0], coors[:,:,1], par, par_flag, shape_coors, shape_flag)

        L2_relative = torch.sqrt(torch.sum((u_pred*flag-u*flag)**2 + (v_pred*flag-v*flag)**2, -1)) / torch.sqrt(torch.sum((u*flag)**2 + (v*flag)**2, -1))
        mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
        num_eval += par.shape[0]

    mean_relative_L2 /= num_eval
    mean_relative_L2 = mean_relative_L2

    return mean_relative_L2

# testing function
def test(model, loader, args, device, num_nodes_list, dir):

    # transforme state to be eval
    model.eval()

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes = num_nodes_list

    mean_relative_L2 = 0
    num_eval = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (par, coors, u, v, flag, par_flag) in loader:

        # extract domain shape information
        if args.geo_node == 'vary_bound' or 'vary_bound_sup':
            ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        if args.geo_node == 'all_bound':
            ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        shape_coors = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

        # prepare the data
        par = par.float().to(device)
        par_flag = par_flag.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)
        flag = flag.float().to(device)

        # model forward
        u_pred, v_pred = model(coors[:,:,0], coors[:,:,1], par, par_flag, shape_coors, shape_flag)

        # compute L2 error
        L2_relative = torch.sqrt(torch.sum((u_pred*flag-u*flag)**2 + (v_pred*flag-v*flag)**2, -1)) / torch.sqrt(torch.sum((u*flag)**2 + (v*flag)**2, -1))
        # L2_relative = torch.sqrt(torch.sum((u_pred*flag-u*flag)**2 , -1)) / torch.sqrt(torch.sum((u*flag)**2, -1))

        # get the prediction that we want
        if dir == 'x':
            pred = u_pred
            gt = u
        if dir == 'y':
            pred = v_pred
            gt = v

        # find the max and min error sample in this batch
        max_err, max_err_idx = torch.topk(L2_relative, 1)
        if max_err > max_relative_err:
            max_relative_err = max_err
            worst_xcoor = coors[max_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_ycoor = coors[max_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
            worst_f = pred[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            worst_gt = gt[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            worst_ff = flag[max_err_idx,:].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(worst_ff>0.5)[0]
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
            best_gt = gt[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            best_ff = flag[min_err_idx,:].squeeze(0).detach().cpu().numpy()
            valid_id = np.where(best_ff>=0.5)[0]
            best_xcoor = best_xcoor[valid_id]
            best_ycoor = best_ycoor[valid_id]
            best_f = best_f[valid_id]
            best_gt = best_gt[valid_id]
            best_ff = best_ff[valid_id]
            

        # compute average error
        mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
        num_eval += par.shape[0]

    mean_relative_L2 /= num_eval
    mean_relative_L2 = mean_relative_L2

    # color bar range
    max_color = np.amax([np.amax(worst_gt), np.amax(best_gt)])
    min_color = np.amin([np.amin(worst_gt), np.amin(best_gt)])

    # errorcolor bar range
    err_max_color = np.amax([np.amax(np.abs(worst_f-worst_gt)), np.amax(np.abs(best_f-best_gt))])

    # make the plot
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15,8))
    plt.subplot(2,3,1)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2,3,2)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.scatter(worst_xcoor, worst_ycoor, c=np.abs(worst_f-worst_gt), cmap=cm, vmin=0, vmax=err_max_color, marker='o', s=3)
    plt.title('absolute error')
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.scatter(best_xcoor, best_ycoor, c=best_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2,3,5)
    plt.scatter(best_xcoor, best_ycoor, c=best_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=3)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.scatter(best_xcoor, best_ycoor, c=np.abs(best_f-best_gt), cmap=cm, vmin=0, vmax=err_max_color, marker='o', s=3)
    plt.title('absolute error')
    plt.colorbar()
    plt.savefig(r'./res/plots/sample_{}_{}_{}_{}.png'.format(args.geo_node, args.model, args.data, dir))

    return mean_relative_L2

# function of extracting the geometry embeddings
def get_geometry_embeddings(model, loader, args, device, num_nodes_list):

    # transforme state to be eval
    model.eval()

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes = num_nodes_list

    # forward to get the embeddings
    all_geo_embeddings = []
    for (par, coors, u, v, flag, par_flag) in loader:

        # extract domain shape information
        if args.geo_node == 'vary_bound' or 'vary_bound_sup':
            ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        if args.geo_node == 'all_bound':
            ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        if args.geo_node == 'all_domain':
            ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
        shape_coors = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
        shape_flag = flag[:, ss_index]
        shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

        # prepare the data
        par = par.float().to(device)
        par_flag = par_flag.float().to(device)
        coors = coors.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)
        flag = flag.float().to(device)

        # model forward
        Geo_embeddings = model.predict_geometry_embedding(coors[:,:,0], coors[:,:,1], 
            par, par_flag, shape_coors, shape_flag)
        all_geo_embeddings.append(Geo_embeddings)
    
    all_geo_embeddings = torch.cat(tuple(all_geo_embeddings), 0)

    return all_geo_embeddings

# define the training function
def train(args, config, model, device, loaders, num_nodes_list, params):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes = num_nodes_list

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
        model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))  
    except:
        print('No trained models') 
    model = model.to(device)

    # define tradeoff weights
    weight_load = config['train']['weight_load']
    weight_pde = config['train']['weight_pde']
    weight_fix = config['train']['weight_fix']
    weight_free = config['train']['weight_free']

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_fix_loss = np.inf
        avg_free_loss = np.inf
        avg_load_loss = np.inf
        for e in pbar:
          
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, args, device, num_nodes_list)
                err_hist.append(err)
                print('Current epoch error:', err)
                print('current epochs pde loss:', avg_pde_loss)
                print('fix bc loss:', avg_fix_loss)
                print('free bc loss:', avg_free_loss)
                print('load bc loss:', avg_load_loss)

                avg_pde_loss = 0
                avg_fix_loss = 0
                avg_free_loss = 0
                avg_load_loss = 0
                if err < min_val_err:
                    torch.save(model.state_dict(), r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            for (par, coors, u, v, flag, par_flag) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # random sampling for PDE residual computation
                    ss_index = np.random.choice(np.arange(max_pde_nodes), config['train']['coor_sampling_size'])
                    pde_sampled_coors = coors[:, ss_index, :]
                    pde_sampled_coors = pde_sampled_coors.float().to(device)    # (B, Ms, 2)
                    pde_flag = flag[:, ss_index]
                    pde_flag = pde_flag.float().to(device)    # (B, Ms)

                    # extract bc loading coordinates
                    ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes)
                    load_coors = coors[:, ss_index, :].float().to(device)    # (B, max_par, 2)
                    load_flag = flag[:, ss_index]
                    load_flag = load_flag.float().to(device)    # (B, max_par)
                    u_load_gt = u[:, ss_index].float().to(device)    # (B, max_par)
                    v_load_gt = v[:, ss_index].float().to(device)    # (B, max_par)

                    # extract bc free condition coordinates
                    ss_index = np.arange(max_pde_nodes + max_par_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes)
                    bcy_coors = coors[:, ss_index, :].float().to(device)    # (B, max_bcy, 2)
                    bcy_flag = flag[:, ss_index]
                    bcy_flag = bcy_flag.float().to(device)    # (B, max_bcy)

                    # extract the fixed condition coordinates
                    ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    bcxy_coors = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
                    bcxy_flag = flag[:, ss_index]
                    bcxy_flag = bcxy_flag.float().to(device)    # (B, max_bcxy)

                    # extract the boundary of the varying shape
                    if args.geo_node == 'vary_bound' or 'vary_bound_sup':
                        ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    if args.geo_node == 'all_bound':
                        ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
                    shape_flag = flag[:, ss_index]
                    shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

                    # prepare the parameter input
                    par = par.float().to(device)    # (B, max_par, 4)
                    par_flag = par_flag.float().to(device)    # (B, max_par)

                    # forward to get the prediction on fixed boundary
                    u_BCxy_pred, v_BCxy_pred = model(bcxy_coors[:,:,0], bcxy_coors[:,:,1], par, par_flag, shape_coor, shape_flag)

                    # forward to get prediction on load boundary
                    u_load_pred, v_load_pred = model(load_coors[:,:,0], load_coors[:,:,1], par, par_flag, shape_coor, shape_flag)
                    
                    # forward to get the prediction on pde domian
                    x_pde = Variable(pde_sampled_coors[:,:,0], requires_grad=True)
                    y_pde = Variable(pde_sampled_coors[:,:,1], requires_grad=True)
                    u_pde_pred, v_pde_pred = model(x_pde, y_pde, par, par_flag, shape_coor, shape_flag)
                    rx, ry = plate_stress_loss(u_pde_pred, v_pde_pred, x_pde, y_pde, params)

                    # forward to get the prediction on free boundary
                    x_pde_bcy = Variable(bcy_coors[:,:,0], requires_grad=True)
                    y_pde_bcy = Variable(bcy_coors[:,:,1], requires_grad=True)
                    u_BCy_pred, v_BCy_pred = model(x_pde_bcy, y_pde_bcy, par, par_flag, shape_coor, shape_flag)
                    sigma_yy, sigma_xy = bc_edgeY_loss(u_BCy_pred, v_BCy_pred, x_pde_bcy, y_pde_bcy, params)

                    # compute the losses
                    pde_loss = torch.mean((rx*pde_flag)**2) + torch.mean((ry*pde_flag)**2)
                    load_loss = mse(u_load_pred*load_flag, u_load_gt*load_flag) + mse(v_load_pred*load_flag, v_load_gt*load_flag)
                    fix_loss = torch.mean((u_BCxy_pred*bcxy_flag)**2) + torch.mean((v_BCxy_pred*bcxy_flag)**2)
                    free_loss = torch.mean((sigma_yy*bcy_flag)**2) + torch.mean((sigma_xy*bcy_flag)**2)
                    total_loss = weight_pde*pde_loss + weight_load*load_loss + weight_fix*fix_loss + weight_free*free_loss

                    # store the loss
                    avg_pde_loss += pde_loss.detach().cpu().item()
                    avg_fix_loss += fix_loss.detach().cpu().item()
                    avg_free_loss = free_loss.detach().cpu().item()
                    avg_load_loss = load_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # clear cuda
                    torch.cuda.empty_cache()

    # final test
    model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))   
    model.eval()
    err = test(model, test_loader, args, device, num_nodes_list, dir='x')
    _ = test(model, test_loader, args, device, num_nodes_list, dir='y')
    print('Best L2 relative error on test loader:', err)


# define the supervised training function
def sup_train(args, config, model, device, loaders, num_nodes_list, params):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes = num_nodes_list

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
        model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))  
    except:
        print('No trained models') 
    model = model.to(device)

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_fix_loss = np.inf
        avg_free_loss = np.inf
        avg_load_loss = np.inf
        for e in pbar:
          
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, args, device, num_nodes_list)
                err_hist.append(err)
                print('Current epoch error:', err)
                print('current epochs pde loss:', avg_pde_loss)
                print('fix bc loss:', avg_fix_loss)
                print('free bc loss:', avg_free_loss)
                print('load bc loss:', avg_load_loss)

                avg_pde_loss = 0
                avg_fix_loss = 0
                avg_free_loss = 0
                avg_load_loss = 0
                if err < min_val_err:
                    torch.save(model.state_dict(), r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            for (par, coors, u, v, flag, par_flag) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # random sampling for PDE residual computation
                    all_coors = coors[:, :, :].float().to(device)
                    all_flag = flag[:, :].float().to(device)    # (B, Ms)
                    u_gt = u.float().to(device) 
                    v_gt = v.float().to(device)

                    # extract the boundary of the varying shape
                    if args.geo_node == 'vary_bound_sup':
                        ss_index = np.arange(max_pde_nodes + max_par_nodes + max_bcy_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    if args.geo_node == 'all_bound':
                        ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0, max_pde_nodes + max_par_nodes + max_bcy_nodes + max_bcxy_nodes)
                    shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
                    shape_flag = flag[:, ss_index]
                    shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

                    # prepare the parameter input
                    par = par.float().to(device)    # (B, max_par, 4)
                    par_flag = par_flag.float().to(device)    # (B, max_par)

                    # forward to get the prediction 
                    u_pred, v_pred = model(all_coors[:,:,0], all_coors[:,:,1], par, par_flag, shape_coor, shape_flag)
                    
                    # compute the loss
                    total_loss = mse(u_pred*all_flag, u_gt*all_flag) + mse(v_pred*all_flag, v_gt*all_flag)

                    # store the loss
                    avg_pde_loss += total_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # clear cuda
                    torch.cuda.empty_cache()

    # final test
    model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model)))   
    model.eval()
    err = test(model, test_loader, args, device, num_nodes_list, dir='x')
    _ = test(model, test_loader, args, device, num_nodes_list, dir='y')
    print('Best L2 relative error on test loader:', err)