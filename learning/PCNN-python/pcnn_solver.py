import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from dataset import Dataset
from pde_cnn import get_Net
import parameters
from Logger import Logger,t_step



class PCNNSolver:
    def __init__(self, dataset: Dataset, params, device):

        #
        # Store local copy of the parameters
        #
        self.params = params

        #
        # Torch device
        #
        self.device = device

        #
        # Dataset
        #
        self.dataset = dataset

        #
        # Torch model
        #
        self.net = get_Net(params).to(self.device)

        #
        # Optimizer
        #
        self.optimizer = Adam(self.net.parameters(), lr=params.lr)

        #
        # Logger
        #
        self.logger = Logger(parameters.get_description(params),use_csv=False,use_tensorboard=params.log)
        if params.load_latest or params.load_date_time is not None or params.load_index is not None:
            self.load_logger = Logger(parameters.get_description(params), use_csv=False, use_tensorboard=False)
            if params.load_optimizer:
                params.load_date_time, params.load_index = self.logger.load_state(self.net, self.optimizer,
                                                                             params.load_date_time, params.load_index)
            else:
                params.load_date_time, params.load_index = self.logger.load_state(self.net, None, params.load_date_time,
                                                                             params.load_index)
            params.load_index = int(params.load_index)
            print(f"loaded: {params.load_date_time}, {params.load_index}")
        params.load_index = 0 if params.load_index is None else params.load_index

        #
        # Convolution kernels
        #

        # First order derivatives
        self.dx_kernel = torch.Tensor([-0.5,0,0.5], device=self.device).view(1, 1, 1, 3)
        self.dx_conv = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1), padding_mode='replicate', bias=False).to(self.device)

        # Copy the weights over to the convolution
        with torch.no_grad():
            self.dx_conv.weight.copy_(self.dx_kernel)
        self.dx_conv.weight.requires_grad_(False)

        self.dy_kernel = torch.Tensor([-0.5,0,0.5], device=self.device).view(1, 1, 3, 1)
        self.dy_conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0), padding_mode='replicate', bias=False).to(self.device)
        with torch.no_grad():
            self.dy_conv.weight.copy_(self.dy_kernel)
        self.dy_conv.weight.requires_grad_(False)

        # Second order derivatives
        self.dx2_kernel = torch.Tensor([1, -2, 1], device=self.device).view(1, 1, 1, 3)
        self.dx2_conv = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1), padding_mode='replicate', bias=False).to(self.device)
        with torch.no_grad():
            self.dx2_conv.weight.copy_(self.dx2_kernel)
        self.dx2_conv.weight.requires_grad_(False)

        self.dy2_kernel = torch.Tensor([1, -2, 1], device=self.device).view(1, 1, 3, 1)
        self.dy2_conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0), padding_mode='replicate', bias=False).to(self.device)
        with torch.no_grad():
            self.dy2_conv.weight.copy_(self.dy2_kernel)
        self.dy2_conv.weight.requires_grad_(False)



    def d_dx(self, quantity):
        return self.dx_conv(quantity) / self.dataset.dx

    def d_dy(self, quantity):
        return self.dy_conv(quantity) / self.dataset.dy

    def d2_dx2(self, quantity):
        return self.dx2_conv(quantity) / (self.dataset.dx**2)

    def d2_dy2(self, quantity):
        return self.dy2_conv(quantity) / (self.dataset.dy**2)

    def train(self):
        """
        Enable training of this solver
        """
        self.net.train()

    def loss_function(self, x):
        return torch.pow(x, 2)

    #
    # Loss functions
    #
    def compute_loss_continuity(self, h_old, h_new, u_new, v_new):
        return (h_new - h_old)/self.params.dt + self.d_dx(u_new * h_new) + self.d_dy(v_new * h_new) - self.params.Hin

    def compute_loss_momentum(self, h_new, u_old, u_new, v_old, v_new, S_new, B_new):

        # Compute bed roughness
        n = self.params.nb + (self.params.nv - self.params.nb) * (B_new / self.params.k)

        # Compute Chezy coefficient using Manning's formulation
        Cz = (1.0 / n) * torch.pow(h_new, 1.0 / 6.0)

        # Bed shear stress components in x and y direction
        bed_shear_stress_precalc = self.params.grav / torch.pow(Cz, 2.0) * torch.pow(
            u_new * u_new + v_new * v_new, 0.5)
        tau_bx_per_rho = bed_shear_stress_precalc * u_new
        tau_by_per_rho = bed_shear_stress_precalc * v_new

        loss_u = (u_new - u_old)/self.params.dt + self.params.grav * self.d_dx(h_new + S_new) + u_new * self.d_dx(u_new) + v_new * self.d_dy(u_new) + tau_bx_per_rho / h_new - self.params.Du * (self.d2_dx2(u_new) + self.d2_dy2(u_new))

        loss_v = (v_new - v_old)/self.params.dt + self.params.grav * self.d_dy(h_new + S_new) + u_new * self.d_dx(v_new) + v_new * self.d_dy(v_new) + tau_by_per_rho / h_new - self.params.Du * (self.d2_dx2(v_new) + self.d2_dy2(v_new))

        return loss_u + loss_v

    def compute_loss_sediment(self, h_new, u_new, v_new, S_old, S_new, B_new):

        # Compute bed roughness
        n = self.params.nb + (self.params.nv - self.params.nb) * (B_new / self.params.k)

        # Compute Chezy coefficient using Manning's formulation
        Cz = (1.0 / n) * torch.pow(h_new, 1.0 / 6.0)

        # The topographic diffusion term requires extra attention
        Ds = self.params.D0 * (1.0 - self.params.pD * (B_new / self.params.k))

        topographic_diffusion_term = self.d_dx(Ds * self.d_dx(S_new)) + self.d_dy(Ds * self.d_dy(S_new))

        # Effective water layer thickness he
        he = h_new - self.params.Hc

        # Compute tau_b_per_rho
        tau_b_per_rho = self.params.grav / torch.pow(Cz, 2.0) * (
                    u_new * u_new + v_new * v_new)

        # Compute dS_dt
        dS_dt = self.params.Sin * (he / (self.params.Qs + he)) - self.params.Es * (1.0 - self.params.pE * (
                    B_new / self.params.k)) * S_new * tau_b_per_rho + topographic_diffusion_term

        dS = dS_dt * self.dt * self.domain.morphological_acc_factor

        S_updated = self.domain.S + dS

        return

    def solve_step(self):
        h, u, v, S, B = self.dataset.ask()

        # Predict the new domain state by performing a forward pass through the network
        h_new, u_new, v_new, S_new, B_new = self.net(h, u, v, S, B)

        # Compute the loss of this update step


        # compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Perform an optimization step
        self.optimizer.step()

        # Recycle the data
        self.dataset.tell(h_new, u_new, v_new, S_new, B_new)


