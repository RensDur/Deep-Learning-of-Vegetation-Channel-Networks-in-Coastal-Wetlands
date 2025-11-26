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

    def loss_function(self, x):
        return torch.pow(x, 2)

    def train(self):
        """
        TRAINING ROUTINE
        """

        # Enable training of the model
        self.net.train()

        # Training loop:
        # Start from the most recently finished epoch and train until the configured number
        # of epochs has been reached.
        for epoch in range(self.params.load_index, self.params.n_epochs):
            # Each epoch consists of a configurable number of batches.
            for i in range(self.params.n_batches_per_epoch):

                # Ask for a batch from the dataset
                h_old, u_old, v_old, S_old, B_old = self.dataset.ask()

                # TODO: MAC grid

                # Predict the new domain state by performing a forward pass through the network
                h_new, u_new, v_new, S_new, B_new = self.net(h_old, u_old, v_old, S_old, B_old)

                # Choose between explicit, implicit or IMEX integration schemes
                if self.params.integrator == "explicit":
                    h = h_old
                    u = u_old
                    v = v_old
                    S = S_old
                    B = B_old
                elif self.params.integrator == "implicit":
                    h = h_new
                    u = u_new
                    v = v_new
                    S = S_new
                    B = B_new
                else: # "imex", default
                    h = (h_new + h_old) / 2
                    u = (u_new + u_old) / 2
                    v = (v_new + v_old) / 2
                    S = (S_new + S_old) / 2
                    B = (B_new + B_old) / 2

                #
                # COMPUTE LOSS
                #

                # 0. Precompute quantities for improved readability
                dh_dt = (h_new - h_old) / self.params.dt
                du_dt = (u_new - u_old) / self.params.dt
                dv_dt = (v_new - v_old) / self.params.dt
                dS_dt = (S_new - S_old) / self.params.dt
                dB_dt = (B_new - B_old) / self.params.dt

                # Compute bed roughness
                n = self.params.nb + (self.params.nv - self.params.nb) * (B / self.params.k)

                # Compute Chezy coefficient using Manning's formulation
                Cz = (1.0 / n) * torch.pow(h, 1.0 / 6.0)

                # Bed shear stress components in x and y direction
                bed_shear_stress_precalc = self.params.grav / torch.pow(Cz, 2.0) * torch.pow(u*u + v*v, 0.5)
                tau_bx_per_rho = bed_shear_stress_precalc * u
                tau_by_per_rho = bed_shear_stress_precalc * v
                tau_b_per_rho = self.params.grav / torch.pow(Cz, 2.0) * (u*u + v*v)

                # The topographic diffusion term requires extra attention
                Ds = self.params.D0 * (1.0 - self.params.pD * (B / self.params.k))

                topographic_diffusion_term = self.d_dx(Ds * self.d_dx(S)) + self.d_dy(Ds * self.d_dy(S))

                # Effective water layer thickness he
                he = h - self.params.Hc

                # 1. Continuity loss
                loss_h = torch.mean(self.loss_function(
                    dh_dt + self.d_dx(u * h) + self.d_dy(v * h) - self.params.Hin
                ), dim=(1,2,3))

                # 2. Momentum loss
                loss_u = du_dt + self.params.grav * self.d_dx(h + S) + u * self.d_dx(u) + v * self.d_dy(u)
                loss_v = dv_dt + self.params.grav * self.d_dy(h + S) + u * self.d_dx(v) + v * self.d_dy(v)

                # Add bed friction effects
                loss_u += tau_bx_per_rho / h
                loss_v += tau_by_per_rho / h

                # Add turbulent mixing effects
                loss_u -= self.params.Du * (self.d2_dx2(u) + self.d2_dy2(u))
                loss_v -= self.params.Du * (self.d2_dx2(v) + self.d2_dy2(v))

                # Apply loss function and compute mean
                loss_u = torch.mean(self.loss_function(loss_u), dim=(1,2,3))
                loss_v = torch.mean(self.loss_function(loss_v), dim=(1,2,3))

                # 3. Sediment loss
                loss_S = torch.mean(self.loss_function(
                    dS_dt \
                    - self.params.Sin * (he / (self.params.Qs + he)) \
                    + self.params.Es * (1.0 - self.params.pE * (B / self.params.k)) * S * tau_b_per_rho \
                    - topographic_diffusion_term
                ), dim=(1,2,3))

                # 4. Vegetation stem density loss
                loss_B = torch.mean(self.loss_function(
                    dB_dt \
                    - self.params.r * B * (1.0 - (B / self.params.k)) * (self.params.Qq / (self.params.Qq + he)) \
                    + self.params.EB * B * tau_b_per_rho \
                    - self.params.DB * (self.d2_dx2(B) + self.d2_dy2(B))
                ), dim=(1,2,3))

                # 5. Boundary loss
                loss_bound = 0

                # Compute combined loss
                loss = torch.mean(torch.log(
                    self.params.loss_h * loss_h \
                    + self.params.loss_momentum * (loss_u + loss_v) \
                    + self.params.loss_S * loss_S \
                    + self.params.loss_B * loss_B \
                    + self.params.loss_bound * loss_bound
                ))

                # Compute gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Perform an optimization step
                self.optimizer.step()

                # TODO: Quantity normalisation? Originally done to normalize pressure and vector potential a

                # Recycle the data
                self.dataset.tell(h_new, u_new, v_new, S_new, B_new)

                # log training metrics
                if i % 10 == 0:
                    loss = loss.cpu().numpy()
                    loss_h = loss_h.cpu().numpy()
                    loss_u = loss_u.cpu().numpy()
                    loss_v = loss_v.cpu().numpy()
                    loss_S = loss_S.cpu().numpy()
                    loss_B = loss_B.cpu().numpy()
                    self.logger.log(f"loss_{self.params.loss}", loss, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_h_{self.params.loss}", loss_h, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_u_{self.params.loss}", loss_u, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_v_{self.params.loss}", loss_v, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_S_{self.params.loss}", loss_S, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_B_{self.params.loss}", loss_B, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_bound_{self.params.loss}", loss_bound, epoch * self.params.n_batches_per_epoch + i)

                    if i % 100 == 0:
                        print(f"{epoch}: i:{i}: loss: {loss}; loss_bound: {loss_bound}; loss_h: {loss_h}; loss_u: {loss_u}; loss_v: {loss_v}; loss_S: {loss_S}; loss_B: {loss_B};")

            # Save the training state after each epoch
            if self.params.log:
                self.logger.save_state(self.net, self.optimizer, epoch + 1)

