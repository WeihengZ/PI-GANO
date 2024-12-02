import torch
import torch.nn as nn
import math

''' ------------------------- baselines -------------------------- '''

# physics-informed DCON
class PI_DCON(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(3, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # trunk network 1
        self.FC1u = nn.Linear(2, config['model']['fc_dim'])
        self.FC2u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC4u = nn.Linear(config['model']['fc_dim_branch'], 1)
        self.act = nn.Tanh()

        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # get the first kernel
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # predict u
        u = self.FC1u(xy)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = torch.mean(u * enc, -1)    # (B, M)

        return u

# physics-informed pointNet
class PI_PN(nn.Module):

    def __init__(self, config):
        super().__init__()

        # encoder network
        trunk_layers = [nn.Linear(3, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.encoder = nn.Sequential(*trunk_layers)

        # decoder network
        trunk_layers = [nn.Linear(2 * config['model']['fc_dim'], config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], 1))
        self.decoder = nn.Sequential(*trunk_layers)
        
    def forward(self, x_coor, y_coor, par, par_flag):
        '''
        par: (B, M)
        x_coor: (B, M)
        y_coor: (B, M)

        return u: (B, M)
        '''

        # get the hidden embeddings
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1), par.unsqueeze(-1)), -1)
        enc = self.encoder(xyf)    # (B, M, F)

        # global feature embeddings
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 1)
        u = pred.squeeze(-1)    # (B, M)

        return u

# physics-informed pointNet for fixed PDE parameters
class PI_PN_only_geo(nn.Module):

    def __init__(self, config):
        super().__init__()

        # encoder network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.encoder = nn.Sequential(*trunk_layers)

        # decoder network
        trunk_layers = [nn.Linear(2 * config['model']['fc_dim'], config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], 1))
        self.decoder = nn.Sequential(*trunk_layers)
        
    def forward(self, x_coor, y_coor, par, par_flag):
        '''
        par: (B, M)
        x_coor: (B, M)
        y_coor: (B, M)

        return u: (B, M)
        '''

        # get the hidden embeddings
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        enc = self.encoder(xyf)    # (B, M, F)

        # global feature embeddings
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 1)
        u = pred.squeeze(-1)    # (B, M)

        return u

''' ------------------------- PI-GANO -------------------------- '''

class DG(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)
        
    def forward(self, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # get the first kernel
        enc = self.branch(shape_coor)    # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)    # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)    # (B, 1, F)

        return Domain_enc

class PI_GANO(nn.Module):

    def __init__(self, config):
        super().__init__()

        # define the geometry encoder
        self.DG = DG(config)

        # branch network
        trunk_layers = [nn.Linear(3, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # trunk network 1
        self.xy_lift = nn.Linear(2, config['model']['fc_dim'])
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,F)

        # combine with global embedding
        xy_global = torch.cat((xy_local, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = torch.mean(u * enc, -1)    # (B, M)
        
        return u

''' ------------------------- study of geometry embedding -------------------------- '''

class DG_other_embedding(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], 2 * config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)
        
    def forward(self, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # get the first kernel
        enc = self.branch(shape_coor)    # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)    # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)    # (B, 1, F)

        return Domain_enc

# Use addition as feature coupling
class PI_GANO_add(nn.Module):

    def __init__(self, config):
        super().__init__()

        
        self.DG = DG_other_embedding(config)

        # branch network
        trunk_layers = [nn.Linear(3, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # trunk network 1
        self.xy_lift = nn.Linear(2, 2*config['model']['fc_dim'])
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,F)

        # combine with global embedding
        xy_global = xy_local + Domain_enc.repeat(1,mD,1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = torch.mean(u * enc, -1)    # (B, M)
        
        return u

# Use elementwise multiplication as feature coupling
class PI_GANO_mul(nn.Module):

    def __init__(self, config):
        super().__init__()

        
        self.DG = DG_other_embedding(config)

        # branch network
        trunk_layers = [nn.Linear(3, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # trunk network 1
        self.xy_lift = nn.Linear(2, 2*config['model']['fc_dim'])
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,F)

        # combine with global embedding
        xy_global = xy_local * Domain_enc.repeat(1,mD,1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = torch.mean(u * enc, -1)    # (B, M)
        
        return u

''' ------------------------- boundary embedding -------------------------- '''

# use high-level feature parameters to represent domain geometry
class PI_GANO_geo(nn.Module):

    def __init__(self, config, geo_feature_dim):
        super().__init__()

        
        # branch network
        trunk_layers = [nn.Linear(geo_feature_dim, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.geo_encoder = nn.Sequential(*trunk_layers)

        # branch network
        trunk_layers = [nn.Linear(3, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # trunk network 1
        self.xy_lift = nn.Linear(2, config['model']['fc_dim'])
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par, par_flag, geo_feature):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.geo_encoder(geo_feature).unsqueeze(1)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,F)

        # combine with global embedding
        xy_global = torch.cat((xy_local, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = torch.mean(u * enc, -1)    # (B, M)
        
        return u

''' ------------------------- New model -------------------------- '''

class New_model_darcy(nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)

        return u: (B, M)
        '''
        
        return None