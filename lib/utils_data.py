import scipy.io as sio
import numpy as np
import torch

# function to generate data loader for darcy problem 
def generate_darcy_data_loader(args, config):

    # load the data
    mat_contents = sio.loadmat(r'./data/{}.mat'.format(args.data))
    u = mat_contents['u_field'][0]    # list of M elements
    coors = mat_contents['coors'][0]    # list of M elements
    par = mat_contents['BC_input_var'][0]    # list of M elements
    ic_flag = mat_contents['totoal_ic_flag'][0]   # list of M elements 

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bc_nodes = 0
    for i in range(datasize):
        num_pde = np.sum(1-ic_flag[i])
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = par[i].shape[0]
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bc = np.sum(ic_flag[i])
        if num_bc > max_bc_nodes:
            max_bc_nodes = num_bc
    max_pde_nodes = int(max_pde_nodes)
    max_bc_nodes = int(max_bc_nodes)
    max_par_nodes = int(max_par_nodes)

    # append zeros to the data 
    uT = []
    coorT = []
    parT = []
    par_flagT = []
    flagT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where(ic_flag[i]==0)[1]
        bc_idx = np.where(ic_flag[i]==1)[1]
        num_pde = np.size(pde_idx)
        num_bc = np.size(bc_idx)
        # re-organize solution
        up = u[i]
        up = np.concatenate((up[:,pde_idx], np.zeros((1,max_pde_nodes-num_pde)), up[:,bc_idx], np.zeros((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        uT.append(up)
        # re-organize coors
        coorp = coors[i]
        coorp = np.concatenate((coorp[pde_idx,:], np.zeros((max_pde_nodes-num_pde,2)), coorp[bc_idx,:], np.zeros((max_bc_nodes-num_bc,2))), 0)    # (max_pde+max_bc,2)
        coorp = np.expand_dims(coorp, 0)    # (1,max_pde+max_bc,2)
        coorT.append(coorp) 
        # re-organize parameters
        parpv = par[i]
        num_par = parpv.shape[0]
        parp = np.concatenate((parpv, np.zeros((max_par_nodes-num_par,3))), 0)    # (max_par,3)
        par_flag = np.concatenate((np.ones_like(parpv), np.zeros((max_par_nodes-num_par,3))), 0)    # (max_par,3)
        parp = np.expand_dims(parp, 0)    # (1,max_par,3)
        par_flag = np.expand_dims(par_flag, 0)    # (1,max_par,3)
        parT.append(parp)
        par_flagT.append(par_flag)
        # re-organize ic flag
        flagp = ic_flag[i]
        flagp = np.concatenate((flagp[:,pde_idx], -np.ones((1,max_pde_nodes-num_pde)), flagp[:,bc_idx], -np.ones((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        flagT.append(flagp)
    uT = np.concatenate(tuple(uT), 0)    # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 2)
    parT = np.concatenate(tuple(parT), 0)    # (M, max_par_nodes,3)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)
    par_flagT = np.concatenate(tuple(par_flagT), 0)[:,:,0]    # (M, max_par_nodes)
    uT = torch.from_numpy(uT)
    coorT = torch.from_numpy(coorT)
    parT = torch.from_numpy(parT)
    flagT = torch.from_numpy(flagT)
    par_flagT = torch.from_numpy(par_flagT)

    # split the data
    bar1 = [0,int(0.7*datasize)]
    bar2 = [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [int(0.8*datasize),int(datasize)]
    train_dataset = torch.utils.data.TensorDataset(parT[bar1[0]:bar1[1],:,:],
            coorT[bar1[0]:bar1[1],:], uT[bar1[0]:bar1[1],:], flagT[bar1[0]:bar1[1],:], par_flagT[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(parT[bar2[0]:bar2[1],:,:], 
            coorT[bar2[0]:bar2[1],:], uT[bar2[0]:bar2[1],:], flagT[bar2[0]:bar2[1],:], par_flagT[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(parT[bar3[0]:bar3[1],:,:], 
            coorT[bar3[0]:bar3[1],:], uT[bar3[0]:bar3[1],:], flagT[bar3[0]:bar3[1],:], par_flagT[bar3[0]:bar3[1],:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batchsize'], shuffle=False)

    # store the number of nodes of different types
    num_nodes_list = (max_pde_nodes, max_bc_nodes, max_par_nodes)

    return train_loader, val_loader, test_loader, num_nodes_list

# function to generate data loader for 2D plate stress problem 
def generate_plate_stress_data_loader(args, config):

    # load the data
    mat_contents = sio.loadmat(r'./data/{}.mat'.format(args.data))
    u = mat_contents['final_u'][0]    # list of M elements
    v = mat_contents['final_v'][0]    # list of M elements
    coors = mat_contents['coors_dict'][0]    # list of M elements
    flag_BC_load = mat_contents['flag_BC_load_dict'][0]
    flag_BCxy = mat_contents['flag_BCxy_dict'][0]
    flag_BCy = mat_contents['flag_BCy_dict'][0]
    nu = mat_contents['poisson'][0][0]    # scalar
    young = mat_contents['young'][0][0]   # scalar
    element_size = mat_contents['element_size'][0][0]   # scalar

    # scale the young's module
    scalar_factor = 1e-4
    young = young * scalar_factor

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bcy_nodes = 0
    max_bcxy_nodes = 0
    for i in range(datasize):
        num_pde = np.sum((1-flag_BC_load[i])*(1-flag_BCxy[i])*(1-flag_BCy[i]))
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = np.sum((flag_BC_load[i]))
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bcy = np.sum(flag_BCy[i])
        if num_bcy > max_bcy_nodes:
            max_bcy_nodes = num_bcy
        num_bcxy = np.sum(flag_BCxy[i])
        if num_bcxy > max_bcxy_nodes:
            max_bcxy_nodes = num_bcxy
    max_pde_nodes = int(max_pde_nodes)
    max_bcxy_nodes = int(max_bcxy_nodes)
    max_bcy_nodes = int(max_bcy_nodes)
    max_par_nodes = int(max_par_nodes)

    # organize the parameters into a list
    par = []
    for i in range(datasize):
        id_param = np.where(flag_BC_load[i]==1)[0]
        params = np.concatenate((coors[i][id_param,:],
                                u[i][id_param,:],
                                v[i][id_param,:]), -1)    # (B, M', 4)
        par.append(params)

    # append zeros to the data 
    uT = []
    vT = []
    coorT = []
    parT = []
    par_flagT = []
    flagT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where((1-flag_BC_load[i])*(1-flag_BCxy[i])*(1-flag_BCy[i])==1)[0]
        bc_load_idx = np.where(flag_BC_load[i]==1)[0]
        bcy_idx = np.where(flag_BCy[i]==1)[0]
        bcxy_idx = np.where(flag_BCxy[i]==1)[0]

        # get the number
        num_pde = np.size(pde_idx)
        num_load = np.size(bc_load_idx)
        num_bcy = np.size(bcy_idx)
        num_bcxy = np.size(bcxy_idx)

        # re-organize solution
        up = u[i]
        up = np.concatenate((
                            up[pde_idx,:], np.zeros((max_pde_nodes-num_pde,1)), 
                            up[bc_load_idx,:], np.zeros((max_par_nodes-num_load,1)),
                            up[bcy_idx,:], np.zeros((max_bcy_nodes-num_bcy,1)),
                            up[bcxy_idx,:], np.zeros((max_bcxy_nodes-num_bcxy,1))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy,1)
        uT.append(up)
        vp = v[i]
        vp = np.concatenate((
                            vp[pde_idx,:], np.zeros((max_pde_nodes-num_pde,1)), 
                            vp[bc_load_idx,:], np.zeros((max_par_nodes-num_load,1)),
                            vp[bcy_idx,:], np.zeros((max_bcy_nodes-num_bcy,1)),
                            vp[bcxy_idx,:], np.zeros((max_bcxy_nodes-num_bcxy,1))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy,1)
        vT.append(vp)

        # re-organize coors
        coorp = coors[i]
        coorp = np.concatenate((
                            coorp[pde_idx,:], np.zeros((max_pde_nodes-num_pde,2)), 
                            coorp[bc_load_idx,:], np.zeros((max_par_nodes-num_load,2)),
                            coorp[bcy_idx,:], np.zeros((max_bcy_nodes-num_bcy,2)),
                            coorp[bcxy_idx,:], np.zeros((max_bcxy_nodes-num_bcxy,2))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy,2)  
        coorp = np.expand_dims(coorp, 0)    # (1, max_pde+max_load+max_bcy+max_bcxy,2) 
        coorT.append(coorp) 

        # re-organize parameters
        parpv = par[i]
        num_par = parpv.shape[0]
        parp = np.concatenate((parpv, np.zeros((max_par_nodes-num_par,4))), 0)    # (max_par,4)
        par_flag = np.concatenate((np.ones_like(parpv), np.zeros((max_par_nodes-num_par,4))), 0)    # (max_par,4)
        parp = np.expand_dims(parp, 0)    # (1,max_par,4)
        par_flag = np.expand_dims(par_flag, 0)    # (1,max_par,4)
        parT.append(parp)
        par_flagT.append(par_flag)

        # re-organize node flags
        flagp = np.concatenate((
                            np.ones_like(pde_idx), np.zeros((max_pde_nodes-num_pde)), 
                            np.ones_like(bc_load_idx), np.zeros((max_par_nodes-num_load)),
                            np.ones_like(bcy_idx), np.zeros((max_bcy_nodes-num_bcy)),
                            np.ones_like(bcxy_idx), np.zeros((max_bcxy_nodes-num_bcxy))
                            ), 0)    # (max_pde+max_load+max_bcy+max_bcxy)  
        flagp = np.expand_dims(flagp, 0) 
        flagT.append(flagp)

    
    uT = np.concatenate(tuple(uT), -1).T    # (M, max_node)
    vT = np.concatenate(tuple(vT), -1).T    # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 2)
    parT = np.concatenate(tuple(parT), 0)    # (M, max_par_nodes,3)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)
    par_flagT = np.concatenate(tuple(par_flagT), 0)[:,:,0]    # (M, max_par_nodes)
    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    coorT = torch.from_numpy(coorT)
    parT = torch.from_numpy(parT)
    flagT = torch.from_numpy(flagT)
    par_flagT = torch.from_numpy(par_flagT)

    print(uT.shape, vT.shape, coorT.shape, parT.shape, flagT.shape, par_flagT.shape)

    # split the data
    bar1 = [0,int(0.7*datasize)]
    bar2 = [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [int(0.8*datasize),int(datasize)]
    train_dataset = torch.utils.data.TensorDataset(
        parT[bar1[0]:bar1[1],:,:],coorT[bar1[0]:bar1[1],:], 
        uT[bar1[0]:bar1[1],:], vT[bar1[0]:bar1[1],:],
        flagT[bar1[0]:bar1[1],:], par_flagT[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(
        parT[bar2[0]:bar2[1],:,:], coorT[bar2[0]:bar2[1],:], 
        uT[bar2[0]:bar2[1],:], vT[bar2[0]:bar2[1],:],
        flagT[bar2[0]:bar2[1],:], par_flagT[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(
        parT[bar3[0]:bar3[1],:,:], coorT[bar3[0]:bar3[1],:], 
        uT[bar3[0]:bar3[1],:], vT[bar3[0]:bar3[1],:],
        flagT[bar3[0]:bar3[1],:], par_flagT[bar3[0]:bar3[1],:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # store the number of nodes of different types
    # store the material properties
    num_nodes_list = (max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes)
    params = (young, nu)

    return train_loader, val_loader, test_loader, num_nodes_list, params