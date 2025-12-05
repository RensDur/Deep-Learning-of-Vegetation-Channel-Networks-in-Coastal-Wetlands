import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from dataset import Dataset
from pde_cnn import get_Net
import parameters
from Logger import Logger,t_step
import cv2
import time
from window import Window



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
        # Convolution kernels
        #

        # First order derivatives
        self.dx_kernel = torch.tensor([-0.5,0,0.5], device=self.device).view(1, 1, 1, 3)
        self.dy_kernel = torch.tensor([-0.5,0,0.5], device=self.device).view(1, 1, 3, 1)

        # Second order derivatives
        self.dx2_kernel = torch.tensor([1.0, -2.0, 1.0], device=self.device).view(1, 1, 1, 3)
        self.dy2_kernel = torch.tensor([1.0, -2.0, 1.0], device=self.device).view(1, 1, 3, 1)

        #
        # MAC-grid convolutions
        #
        self.mean_left_kernel = torch.tensor([0.5, 0.5, 0], device=self.device).view(1, 1, 1, 3)
        self.mean_right_kernel = torch.tensor([0, 0.5, 0.5], device=self.device).view(1, 1, 1, 3)
        self.mean_top_kernel = torch.tensor([0.5, 0.5, 0], device=self.device).view(1, 1, 3, 1)
        self.mean_bottom_kernel = torch.tensor([0, 0.5, 0.5], device=self.device).view(1, 1, 3, 1)



    def d_dx(self, quantity):
        return F.conv2d(quantity, self.dx_kernel, padding=(0,1)) / self.dataset.dx

    def d_dy(self, quantity):
        return F.conv2d(quantity, self.dy_kernel, padding=(1,0)) / self.dataset.dy

    def d2_dx2(self, quantity):
        return F.conv2d(quantity, self.dx2_kernel, padding=(0,1)) / (self.dataset.dx**2)

    def d2_dy2(self, quantity):
        return F.conv2d(quantity, self.dy2_kernel, padding=(1,0)) / (self.dataset.dy**2)

    def mac_mean_left(self, quantity):
        return F.conv2d(quantity, self.mean_left_kernel, padding=(0,1))

    def mac_mean_right(self, quantity):
        return F.conv2d(quantity, self.mean_right_kernel, padding=(0,1))

    def mac_mean_top(self, quantity):
        return F.conv2d(quantity, self.mean_top_kernel, padding=(1,0))

    def mac_mean_bottom(self, quantity):
        return F.conv2d(quantity, self.mean_bottom_kernel, padding=(1,0))

    def staggered2normal(self, u, v):
        u = self.mac_mean_left(u)
        v = self.mac_mean_top(v)
        return u, v

    def normal2staggered(self, u, v):
        u = self.mac_mean_right(u)
        v = self.mac_mean_bottom(v)
        return u, v

    def loss_function(self, x):
        return torch.pow(x, 2)

    def train(self):
        """
        TRAINING ROUTINE
        """

        # Initialize randomization seeds
        torch.manual_seed(0)
        np.random.seed(0)

        #
        # Optimizer
        #
        self.optimizer = Adam(self.net.parameters(), lr=self.params.lr)

        #
        # Logger
        #
        self.logger = Logger(parameters.get_description(self.params), use_csv=False, use_tensorboard=self.params.log)
        if self.params.load_latest or self.params.load_date_time is not None or self.params.load_index is not None:
            self.load_logger = Logger(parameters.get_description(self.params), use_csv=False, use_tensorboard=False)
            if self.params.load_optimizer:
                self.params.load_date_time, self.params.load_index = self.logger.load_state(self.net, self.optimizer,
                                                                                  self.params.load_date_time,
                                                                                  self.params.load_index)
            else:
                self.params.load_date_time, self.params.load_index = self.logger.load_state(self.net, None, self.params.load_date_time,
                                                                                  self.params.load_index)
            self.params.load_index = int(self.params.load_index)
            print(f"loaded: {self.params.load_date_time}, {self.params.load_index}")
        self.params.load_index = 0 if self.params.load_index is None else self.params.load_index

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

                # Closed boundary on the left
                loss_bound_left = torch.mean(self.loss_function(
                    u[:, :, :, 0] + u[:, :, :, 1]
                ))
                loss_bound_left += torch.mean(self.loss_function(
                    v[:, :, :, 0] - v[:, :, :, 1]
                ))
                loss_bound_left += torch.mean(self.loss_function(
                    h[:, :, :, 0] - h[:, :, :, 1]
                ))
                loss_bound_left += torch.mean(self.loss_function(
                    S[:, :, :, 0] - S[:, :, :, 1]
                ))
                loss_bound_left += torch.mean(self.loss_function(
                    B[:, :, :, 0] - B[:, :, :, 1]
                ))

                # # Closed boundary on the right
                # loss_bound_right = torch.mean(self.loss_function(
                #     u[:, :, :, -1] + u[:, :, :, -2]
                # ))
                # loss_bound_right += torch.mean(self.loss_function(
                #     v[:, :, :, -1] - v[:, :, :, -2]
                # ))
                # loss_bound_right += torch.mean(self.loss_function(
                #     h[:, :, :, -1] - h[:, :, :, -2]
                # ))
                # loss_bound_right += torch.mean(self.loss_function(
                #     S[:, :, :, -1] - S[:, :, :, -2]
                # ))
                # loss_bound_right += torch.mean(self.loss_function(
                #     B[:, :, :, -1] - B[:, :, :, -2]
                # ))

                # Open boundary on the right
                loss_bound_right = torch.mean(self.loss_function(
                    u[:, :, :, -1] - 2*u[:, :, :, -2] + u[:, :, :, -3]
                ))
                loss_bound_right += torch.mean(self.loss_function(
                    v[:, :, :, -1] - 2*v[:, :, :, -2] + v[:, :, :, -3]
                ))
                loss_bound_right += torch.mean(self.loss_function(
                    h[:, :, :, -1] - h[:, :, :, -2]
                ))
                loss_bound_right += torch.mean(self.loss_function(
                    S[:, :, :, -1]
                ))
                loss_bound_right += torch.mean(self.loss_function(
                    B[:, :, :, -1] - B[:, :, :, -2]
                ))

                # Closed boundary at the top
                loss_bound_top = torch.mean(self.loss_function(
                    u[:, :, 0, :] - u[:, :, 1, :]
                ))
                loss_bound_top += torch.mean(self.loss_function(
                    v[:, :, 0, :] + v[:, :, 1, :]
                ))
                loss_bound_top += torch.mean(self.loss_function(
                    h[:, :, 0, :] - h[:, :, 1, :]
                ))
                loss_bound_top += torch.mean(self.loss_function(
                    S[:, :, 0, :] - S[:, :, 1, :]
                ))
                loss_bound_top += torch.mean(self.loss_function(
                    B[:, :, 0, :] - B[:, :, 1, :]
                ))

                # Closed boundary at the bottom
                loss_bound_bottom = torch.mean(self.loss_function(
                    u[:, :, -1, :] - u[:, :, -2, :]
                ))
                loss_bound_bottom += torch.mean(self.loss_function(
                    v[:, :, -1, :] + v[:, :, -2, :]
                ))
                loss_bound_bottom += torch.mean(self.loss_function(
                    h[:, :, -1, :] - h[:, :, -2, :]
                ))
                loss_bound_bottom += torch.mean(self.loss_function(
                    S[:, :, -1, :] - S[:, :, -2, :]
                ))
                loss_bound_bottom += torch.mean(self.loss_function(
                    B[:, :, -1, :] - B[:, :, -2, :]
                ))

                # Final boundary loss
                loss_bound = loss_bound_left + loss_bound_right + loss_bound_top + loss_bound_bottom

                #
                # Regularizers
                #
                # loss_reg = torch.mean(self.loss_function(
                #     (torch.sum(h_new, dim=(1,2,3)) - torch.sum(h_old, dim=(1,2,3))) / (self.params.width * self.params.height)
                # ))

                # Compute combined loss
                loss = torch.mean(torch.log(
                    (self.params.loss_h * loss_h
                    + self.params.loss_momentum * (loss_u + loss_v)
                    + self.params.loss_S * loss_S
                    + self.params.loss_B * loss_B
                    + self.params.loss_bound * loss_bound
                    # + self.params.loss_reg * loss_reg
                    ) * self.params.loss_multiplier
                ))

                # Compute gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Perform an optimization step
                self.optimizer.step()

                # Recycle the data
                self.dataset.tell(h_new, u_new, v_new, S_new, B_new, random_reset=True)

                # log training metrics
                if i % 10 == 0:
                    loss = float(loss.detach().cpu().numpy())
                    loss_h = float(torch.mean(loss_h).detach().cpu().numpy())
                    loss_u = float(torch.mean(loss_u).detach().cpu().numpy())
                    loss_v = float(torch.mean(loss_v).detach().cpu().numpy())
                    loss_S = float(torch.mean(loss_S).detach().cpu().numpy())
                    loss_B = float(torch.mean(loss_B).detach().cpu().numpy())
                    loss_bound = float(torch.mean(loss_bound).detach().cpu().numpy())
                    # loss_reg = float(torch.mean(loss_reg).detach().cpu().numpy())
                    self.logger.log(f"loss_{self.params.loss}", loss, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_h_{self.params.loss}", loss_h, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_u_{self.params.loss}", loss_u, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_v_{self.params.loss}", loss_v, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_S_{self.params.loss}", loss_S, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_B_{self.params.loss}", loss_B, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_bound_{self.params.loss}", loss_bound, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_reg_{self.params.loss}", loss_reg, epoch * self.params.n_batches_per_epoch + i)

                    print(f"Epoch {epoch}, iter.{i}:\tloss: {round(loss,2)};\tloss_bound: {round(loss_bound,2)};\tloss_h: {round(loss_h,2)};\tloss_u: {round(loss_u,2)};\tloss_v: {round(loss_v,2)};\tloss_S: {round(loss_S,2)};\tloss_B: {round(loss_B,2)}; \tvRAM allocated: {round(torch.mps.current_allocated_memory()/1000000000.0, 2)}GB")
                    # if i % 100 == 0:

            # Save the training state after each epoch
            if self.params.log:
                self.logger.save_state(self.net, self.optimizer, epoch + 1)




    def visualize(self):
        """
        VISUALIZING RESULTS
        """

        # Initialize randomization seeds
        torch.manual_seed(1)
        np.random.seed(6)

        #
        # Logger
        #
        self.logger = Logger(parameters.get_description(self.params), use_csv=False, use_tensorboard=False)

        # Load the trained model state
        date_time, index = self.logger.load_state(self.net, None, datetime=self.params.load_date_time, index=self.params.load_index)

        # Enable evaluation of the model
        self.net.eval()

        print(f"Loaded {self.params.net}: {date_time}, index: {index}")

        # Open a visualization window
        win = Window("Sediment Bed", self.params.width, self.params.height)
        win.set_data_range(0, 1)

        with torch.no_grad():

            # Simulation loop
            while win.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old, S_old, B_old = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                h_new, u_new, v_new, S_new, B_new = self.net(h_old, u_old, v_old, S_old, B_old)

                # Display water level thickness h
                win.put_image(S_new[0, 0].clone().detach().cpu().numpy())
                win.update()

                # Store the newly obtained result in the dataset
                self.dataset.tell(h_new, u_new, v_new, S_new, B_new, random_reset=False)


    def visualize_numerical(self):
        """
        VISUALIZING NUMERICAL REFERENCE SIMULATION
        """

        # Initialize randomization seeds
        torch.manual_seed(1)
        np.random.seed(6)

        # Open a visualization window
        win = Window("Sediment Bed", self.params.width, self.params.height)
        win.set_data_range(0, 1)

        with torch.no_grad():

            # Simulation loop
            while win.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old, S_old, B_old = self.dataset.ask()

                # Predict the new domain state by numerical simulation
                h = h_old
                u = u_old
                v = v_old
                S = S_old
                B = B_old

                #
                # MOMENTUM EQUATIONS
                #

                # Compute bed roughness
                n = self.params.nb + (self.params.nv - self.params.nb) * (B / self.params.k)

                # Compute Chezy coefficient using Manning's formulation
                Cz = (1.0 / n) * torch.pow(h, 1.0/6.0)

                # Bed shear stress components in x and y direction
                bed_shear_stress_precalc = self.params.grav/torch.pow(Cz, 2.0) * torch.pow(u * u + v * v, 0.5)
                tau_bx_per_rho = bed_shear_stress_precalc * u
                tau_by_per_rho = bed_shear_stress_precalc * v

                # Base momentum parts
                du_dt = -self.params.grav * self.d_dx(h + S) - u * self.d_dx(u) - v * self.d_dy(u)
                dv_dt = -self.params.grav * self.d_dy(h + S) - u * self.d_dx(v) - v * self.d_dy(v)

                # Add effects of bed friction
                du_dt -= tau_bx_per_rho / h
                dv_dt -= tau_by_per_rho / h

                # Add effects of turbulent mixing
                du_dt += self.params.Du * (self.d2_dx2(u) + self.d2_dy2(u))
                dv_dt += self.params.Du * (self.d2_dx2(v) + self.d2_dy2(v))

                u += du_dt * self.params.dt
                v += dv_dt * self.params.dt

                #
                # Enforce boundary conditions on u and v
                #

                # Left boundary
                u[:, :, :, 0] = -u[:, :, :, 1]
                v[:, :, :, 0] = v[:, :, :, 1]

                # Right boundary: open outflow
                u[:, :, :, -1] = 2*u[:, :, :, -2] - u[:, :, :, -3]
                v[:, :, :, -1] = 2*v[:, :, :, -2] - v[:, :, :, -3]

                # Top
                u[:, :, 0, :] = u[:, :, 1, :]
                v[:, :, 0, :] = -v[:, :, 1, :]

                # Bottom
                u[:, :, -1, :] = u[:, :, -2, :]
                v[:, :, -1, :] = -v[:, :, -2, :]

                #
                # WATER LAYER THICKNESS
                #

                dh_dt = - self.d_dx(u * h) - self.d_dy(v * h) + self.params.Hin
                h_new = torch.clamp(h + dh_dt * self.params.dt, min=self.params.Hc)

                #
                # SEDIMENTARY BED ELEVATION S
                #

                # The topographic diffusion term requires extra attention
                Ds = self.params.D0 * (1.0 - self.params.pD * (B / self.params.k))

                topographic_diffusion_term = self.d_dx(Ds * self.d_dx(S)) + self.d_dy(Ds * self.d_dy(S))

                # Effective water layer thickness he
                he = h - self.params.Hc

                # Compute tau_b_per_rho
                tau_b_per_rho = self.params.grav/torch.pow(Cz, 2.0) * (u*u + v*v)

                # Compute dS_dt
                dS_dt = self.params.Sin * (he / (self.params.Qs + he)) - self.params.Es*(1.0 - self.params.pE * (B / self.params.k)) * S * tau_b_per_rho + topographic_diffusion_term
                dS = dS_dt * self.params.dt * self.params.morphological_acc_factor
                S_new = S + dS

                #
                # VEGETATION STEM DENSITY B
                #

                dB_dt = self.params.r * B * (1.0 - (B / self.params.k))*(self.params.Qq / (self.params.Qq + he)) \
                        - self.params.EB * B * tau_b_per_rho + self.params.DB * (self.d2_dx2(B) + self.d2_dy2(B))

                dB = dB_dt * self.params.dt * self.params.morphological_acc_factor
                B_new = B + dB

                #
                # Now update the dataset in one go
                #
                h = h_new
                S = S_new
                B = B_new

                #
                # Enforce boundary conditions on h, S and B
                #

                # Left
                h[:, :, :, 0] = h[:, :, :, 1]
                S[:, :, :, 0] = S[:, :, :, 1]
                B[:, :, :, 0] = B[:, :, :, 1]

                # Right: open outflow
                h[:, :, :, -1] = h[:, :, :, -2]
                S[:, :, :, -1] = 0 # Due to balance in erosion and sedimentation on the open boundary, always zero
                B[:, :, :, -1] = B[:, :, :, -2]

                # Top
                h[:, :, 0, :] = h[:, :, 1, :]
                S[:, :, 0, :] = S[:, :, 1, :]
                B[:, :, 0, :] = B[:, :, 1, :]

                # Bottom
                h[:, :, -1, :] = h[:, :, -2, :]
                S[:, :, -1, :] = S[:, :, -2, :]
                B[:, :, -1, :] = B[:, :, -2, :]

                # Display water level thickness h
                win.put_image(S[0, 0].clone().detach().cpu().numpy())
                win.update()

                # Store the newly obtained result in the dataset
                self.dataset.tell(h, u, v, S, B, random_reset=False)

