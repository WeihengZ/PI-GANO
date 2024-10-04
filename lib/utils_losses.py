import torch.nn as nn
import torch

# define darcy loss
def darcy_loss(u, x_coor, y_coor, flag_pde):

    # define loss
    mse = nn.MSELoss()

    # compute pde residual
    u_x = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    pde_residual = u_xx + u_yy + 10
    pde_loss = mse(pde_residual*flag_pde, torch.zeros_like(pde_residual))

    return pde_loss

# PINO loss
def plate_stress_loss(u, v, x_coor, y_coor, params):

    # extract parameters
    E, mu = params
    G = E / 2 / (1+mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    eps_xy = (u_y + v_x)

    # compute stress
    sigma_xx = (E / (1-mu**2)) * (eps_xx + mu*(eps_yy))
    sigma_yy = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sigma_xy = G * eps_xy

    # compute residual
    rx = torch.autograd.grad(outputs=sigma_xx, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx),create_graph=True)[0] +\
         torch.autograd.grad(outputs=sigma_xy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy),create_graph=True)[0]
    ry = torch.autograd.grad(outputs=sigma_xy, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx),create_graph=True)[0] +\
         torch.autograd.grad(outputs=sigma_yy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy),create_graph=True)[0]

    return rx, ry

def bc_edgeY_loss(u, v, x_coor, y_coor, params):

    # extract parameters
    E, mu = params
    G = E / 2 / (1+mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    eps_xy = (u_y + v_x)
    
    # compute stress
    sigma_yy = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sigma_xy = G * eps_xy

    return sigma_yy, sigma_xy