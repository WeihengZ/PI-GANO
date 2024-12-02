import torch
import torch.nn as nn
import math

''' ------------------------- baselines -------------------------- '''
# physics-informed DCON
class DCON(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(4, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # trunk network 1
        self.FC1u = nn.Linear(2, config['model']['fc_dim'])
        self.FC2u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC4u = nn.Linear(config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2, config['model']['fc_dim'])
        self.FC2v = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3v = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC4v = nn.Linear(config['model']['fc_dim'], 1)

        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: not used
        shape_flag: not used

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

        # predict v
        v = self.FC1v(xy)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)   # (B,M,F)
        v = torch.mean(v * enc, -1)    # (B, M)

        return u, v

# physics-informed pointNet
class PIPN(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(4, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.encoder = nn.Sequential(*trunk_layers)

        trunk_layers = [nn.Linear(2 * config['model']['fc_dim'], config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], 2))
        self.decoder = nn.Sequential(*trunk_layers)

    def forward(self, x_coor, y_coor, u_input, v_input, flag):
        '''
        x_coor: (B, M)
        y_coor: (B, M)
        u_input: (B, M)
        v_input: (B, M)
        flag: (B, M)

        return u: (B, M)
        '''

        # construct inputs
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1), u_input.unsqueeze(-1), v_input.unsqueeze(-1)), -1)

        # get the first kernel
        enc = self.encoder(xyf)    # (B, M, F)
        enc_masked = enc * flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # combine
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 2)
        u = pred[:,:,0]
        v = pred[:,:,1]

        return u, v

# physics-informed pointNet for fixed PDE parameters
class PIPN_only_geo(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.encoder = nn.Sequential(*trunk_layers)

        trunk_layers = [nn.Linear(2 * config['model']['fc_dim'], config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], 2))
        self.decoder = nn.Sequential(*trunk_layers)

    def forward(self, x_coor, y_coor, flag):
        '''
        x_coor: (B, M)
        y_coor: (B, M)
        u_input: (B, M)
        v_input: (B, M)
        flag: (B, M)

        return u: (B, M)
        '''

        # construct inputs
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # get the first kernel
        enc = self.encoder(xyf)    # (B, M, F)
        enc_masked = enc * flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # combine
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 2)
        u = pred[:,:,0]
        v = pred[:,:,1]

        return u, v

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
        shape_coor: (B, M'', 2)
        shape_flag: (B, M'')

        return u: (B, 1, F)
        '''

        # get the first kernel
        enc = self.branch(shape_coor)    # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)    # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)    # (B, 1, F)

        return Domain_enc

class GANO(nn.Module):

    def __init__(self, config):
        super().__init__()

        # define the geometry encoder
        self.DG = DG(config)

        # branch network
        trunk_layers = [nn.Linear(4, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5v = nn.Linear(2*config['model']['fc_dim'], 1)
    
    def predict_geometry_embedding(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):

        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        return Domain_enc
  
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # combine with global embedding
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global_u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = self.act(u)
        # u = u * enc
        u = self.FC4u(u)   # (B,M,F)
        # u = self.act(u)
        u = torch.mean(u * enc, -1)    # (B, M)

        # predict v
        v = self.FC1v(xy_global_v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)   # (B,M,F)
        v = self.act(v)
        # v = v * enc
        v = self.FC4v(v)   # (B,M,F)
        # v = self.act(v)
        v = torch.mean(v * enc, -1)    # (B, M)
        
        return u, v


''' ------------------------- parameteric geometry embedding -------------------------- '''

# use high-level feature parameters to represent domain geometry
class GANO_geo(nn.Module):

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
        trunk_layers = [nn.Linear(4, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5v = nn.Linear(2*config['model']['fc_dim'], 1)
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag, geo_feature):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.geo_encoder(geo_feature).unsqueeze(1)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # combine with global embedding
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global_u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = self.act(u)
        # u = u * enc
        u = self.FC4u(u)   # (B,M,F)
        # u = self.act(u)
        u = torch.mean(u * enc, -1)    # (B, M)

        # predict v
        v = self.FC1v(xy_global_v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)   # (B,M,F)
        v = self.act(v)
        # v = v * enc
        v = self.FC4v(v)   # (B,M,F)
        # v = self.act(v)
        v = torch.mean(v * enc, -1)    # (B, M)
        
        return u, v

class baseline_geo(nn.Module):

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
        trunk_layers = [nn.Linear(4, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5v = nn.Linear(2*config['model']['fc_dim'], 1)
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag, geo_feature):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.geo_encoder(geo_feature).unsqueeze(1)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # combine with global embedding
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global_u)   # (B,M,F)
        u = self.act(u)
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = self.FC3u(u)   # (B,M,F)
        u = self.act(u)
        u = self.FC4u(u)   # (B,M,F)

        # predict v
        v = self.FC1v(xy_global_v)   # (B,M,F)
        v = self.act(v)
        v = self.FC2v(v)   # (B,M,F)
        v = self.act(v)
        v = self.FC3v(v)   # (B,M,F)
        v = self.act(v)
        v = self.FC4v(v)   # (B,M,F)
        
        return u, v

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
        shape_coor: (B, M'', 2)
        shape_flag: (B, M'')

        return u: (B, 1, F)
        '''

        # get the first kernel
        enc = self.branch(shape_coor)    # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)    # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)    # (B, 1, F)

        return Domain_enc

# Use addition as feature coupling
class GANO_add(nn.Module):

    def __init__(self, config):
        super().__init__()

        
        self.DG = DG_other_embedding(config)

        # branch network
        trunk_layers = [nn.Linear(4, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, 2*config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, 2*config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5v = nn.Linear(2*config['model']['fc_dim'], 1)
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # combine with global embedding
        xy_global_u = xy_local_u + Domain_enc.repeat(1,mD,1)    # (B,M,2F)
        xy_global_v = xy_local_v + Domain_enc.repeat(1,mD,1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global_u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = self.act(u)
        # u = u * enc
        u = self.FC4u(u)   # (B,M,F)
        # u = self.act(u)
        u = torch.mean(u * enc, -1)    # (B, M)

        # predict v
        v = self.FC1v(xy_global_v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)   # (B,M,F)
        v = self.act(v)
        # v = v * enc
        v = self.FC4v(v)   # (B,M,F)
        # v = self.act(v)
        v = torch.mean(v * enc, -1)    # (B, M)
        
        return u, v

# Use elementwise multiplication as feature coupling
class GANO_mul(nn.Module):

    def __init__(self, config):
        super().__init__()

        
        self.DG = DG_other_embedding(config)

        # branch network
        trunk_layers = [nn.Linear(4, 2*config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, 2*config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, 2*config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4u = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5u = nn.Linear(2*config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC2v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC3v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC4v = nn.Linear(2*config['model']['fc_dim'], 2*config['model']['fc_dim'])
        self.FC5v = nn.Linear(2*config['model']['fc_dim'], 1)
        
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)    # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # combine with global embedding
        xy_global_u = xy_local_u * Domain_enc.repeat(1,mD,1)    # (B,M,2F)
        xy_global_v = xy_local_v * Domain_enc.repeat(1,mD,1)    # (B,M,2F)

        # get the kernels
        enc = self.branch(par)    # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global_u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   # (B,M,F)
        u = self.act(u)
        # u = u * enc
        u = self.FC4u(u)   # (B,M,F)
        # u = self.act(u)
        u = torch.mean(u * enc, -1)    # (B, M)

        # predict v
        v = self.FC1v(xy_global_v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)   # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)   # (B,M,F)
        v = self.act(v)
        # v = v * enc
        v = self.FC4v(v)   # (B,M,F)
        # v = self.act(v)
        v = torch.mean(v * enc, -1)    # (B, M)
        
        return u, v


''' ------------------------- New model -------------------------------------------- '''

class New_model_plate(nn.Module):

    def __init__(self, config):
        super().__init__()

  
    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u, v: (B, M)
        '''
        u = None
        v = None

        return u, v