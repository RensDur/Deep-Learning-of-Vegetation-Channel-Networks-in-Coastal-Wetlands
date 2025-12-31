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
import matplotlib.pyplot as plt



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

        # Open a matplot window - if plotting is enabled
        if self.params.plot_loss:
            plt.ion()

            plot_fig, plot_axs = plt.subplots(1, 3, figsize=(20, 10))

            # Plots
            plot_axs[0].set(title="Loss image", xlabel="x", ylabel="y")
            plot_axs[1].set(title="Total loss", xlabel="x", ylabel="y")
            plot_axs[2].set(title="Loss terms", xlabel="x", ylabel="y")

            # plot_axs[0].grid()
            # plot_axs[0].grid(which="minor", color="0.5")
            # plot_axs[1].grid()
            # plot_axs[1].grid(which="minor", color="0.5")
            # plot_axs[2].grid()
            # plot_axs[2].grid(which="minor", color="0.5")

            # Leftmost plot shows loss image
            plot_loss_image = plot_axs[0].imshow(np.zeros((self.params.height, self.params.width)), cmap="gray", vmin=0, vmax=1)

            # Middle plot shows total loss over time
            plot_loss_total_data = np.array([])

            # Rightmost plot shows loss-terms over time (multiplied by scaling)
            plot_loss_h_data = np.array([])
            plot_loss_momentum_data = np.array([])
            plot_loss_bound_data = np.array([])
            plot_loss_reg_data = np.array([])

            plot_loss_total_graph = plot_axs[1].plot(range(plot_loss_total_data.shape[0]), plot_loss_total_data)[0]

            plot_loss_h_graph = plot_axs[2].plot(range(plot_loss_h_data.shape[0]), plot_loss_h_data, label="h-loss")[0]
            plot_loss_momentum_graph = plot_axs[2].plot(range(plot_loss_momentum_data.shape[0]), plot_loss_momentum_data, label="u,v-loss")[0]
            plot_loss_bound_graph = plot_axs[2].plot(range(plot_loss_bound_data.shape[0]), plot_loss_bound_data, label="bound-loss")[0]
            plot_loss_reg_graph = plot_axs[2].plot(range(plot_loss_reg_data.shape[0]), plot_loss_reg_data, label="reg-loss")[0]

            plot_axs[2].legend(handles=[plot_loss_h_graph, plot_loss_momentum_graph, plot_loss_bound_graph, plot_loss_reg_graph], loc="upper right")

            plt.show()


        # Training loop:
        # Start from the most recently finished epoch and train until the configured number
        # of epochs has been reached.
        for epoch in range(self.params.load_index, self.params.n_epochs):
            # Each epoch consists of a configurable number of batches.
            for i in range(self.params.n_batches_per_epoch):

                # Ask for a batch from the dataset
                h_old, u_old, v_old, cond_mask, flux_x_cond, flux_y_cond = self.dataset.ask()

                # The flow mask is simply 1-cond_mask
                flow_mask = 1 - cond_mask

                # # Convert u,v_cond, cond_mask and flow_mask to MAC grid
                # flux_x_cond, flux_y_cond = self.normal2staggered(flux_x_cond, flux_y_cond)
                # cond_mask_mac = (self.normal2staggered(cond_mask.repeat(1,2,1,1))==1).float()
                # flow_mask_mac = (self.normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()

                # Predict the new domain state by performing a forward pass through the network
                flux_x, flux_y = self.net(h_old, u_old, v_old, cond_mask, flow_mask, flux_x_cond, flux_y_cond)

                dh_dt = - self.d_dx(flux_x) - self.d_dy(flux_y)

                h_new = h_old + dh_dt * self.params.dt

                h_new = torch.clamp(h_new, min=self.params.Hc)

                u_new = flux_x / h_new
                v_new = flux_y / h_new

                # Choose between explicit, implicit or IMEX integration schemes
                if self.params.integrator == "explicit":
                    h = h_old
                    u = u_old
                    v = v_old
                    # S = S_old
                    # B = B_old
                elif self.params.integrator == "implicit":
                    h = h_new
                    u = u_new
                    v = v_new
                    # S = S_new
                    # B = B_new
                else: # "imex", default
                    h = (h_new + h_old) / 2
                    u = (u_new + u_old) / 2
                    v = (v_new + v_old) / 2
                    # S = (S_new + S_old) / 2
                    # B = (B_new + B_old) / 2

                # TODO: REMOVE!!
                S = torch.zeros_like(h).to(self.device)
                B = torch.zeros_like(h).to(self.device)

                #
                # COMPUTE LOSS
                #

                # 0. Precompute quantities for improved readability
                dh_dt = (h_new - h_old) / self.params.dt
                du_dt = (u_new - u_old) / self.params.dt
                dv_dt = (v_new - v_old) / self.params.dt
                # dS_dt = (S_new - S_old) / self.params.dt
                # dB_dt = (B_new - B_old) / self.params.dt

                # # Compute bed roughness
                # n = self.params.nb + (self.params.nv - self.params.nb) * (B / self.params.k)
                #
                # # Compute Chezy coefficient using Manning's formulation
                # Cz = (1.0 / n) * torch.pow(h, 1.0 / 6.0)
                #
                # # Bed shear stress components in x and y direction
                # bed_shear_stress_precalc = self.params.grav / torch.pow(Cz, 2.0) * torch.pow(u*u + v*v, 0.5)
                # tau_bx_per_rho = bed_shear_stress_precalc * u
                # tau_by_per_rho = bed_shear_stress_precalc * v
                # tau_b_per_rho = self.params.grav / torch.pow(Cz, 2.0) * (u*u + v*v)
                #
                # # The topographic diffusion term requires extra attention
                # Ds = self.params.D0 * (1.0 - self.params.pD * (B / self.params.k))
                #
                # topographic_diffusion_term = self.d_dx(Ds * self.d_dx(S)) + self.d_dy(Ds * self.d_dy(S))
                #
                # # Effective water layer thickness he
                # he = h - self.params.Hc

                # 1. Continuity loss
                loss_h = torch.mean(self.loss_function(
                    flow_mask * (dh_dt + self.d_dx(u * h) + self.d_dy(v * h)) # - self.params.Hin
                ), dim=1)

                # 2. Momentum loss
                loss_u = du_dt + self.params.grav * self.d_dx(h + S) + u * self.d_dx(u) + v * self.d_dy(u)
                loss_v = dv_dt + self.params.grav * self.d_dy(h + S) + u * self.d_dx(v) + v * self.d_dy(v)

                # # Add bed friction effects
                # loss_u += tau_bx_per_rho / h
                # loss_v += tau_by_per_rho / h
                #
                # # Add turbulent mixing effects
                loss_u -= self.params.Du * (self.d2_dx2(u) + self.d2_dy2(u))
                loss_v -= self.params.Du * (self.d2_dx2(v) + self.d2_dy2(v))

                # Apply loss function and compute mean
                loss_u = torch.mean(self.loss_function(flow_mask * loss_u), dim=1)
                loss_v = torch.mean(self.loss_function(flow_mask * loss_v), dim=1)

                # # 3. Sediment loss
                # loss_S = torch.mean(self.loss_function(
                #     dS_dt \
                #     - self.params.Sin * (he / (self.params.Qs + he)) \
                #     + self.params.Es * (1.0 - self.params.pE * (B / self.params.k)) * S * tau_b_per_rho \
                #     - topographic_diffusion_term
                # ), dim=0)
                #
                # # 4. Vegetation stem density loss
                # loss_B = torch.mean(self.loss_function(
                #     dB_dt \
                #     - self.params.r * B * (1.0 - (B / self.params.k)) * (self.params.Qq / (self.params.Qq + he)) \
                #     + self.params.EB * B * tau_b_per_rho \
                #     - self.params.DB * (self.d2_dx2(B) + self.d2_dy2(B))
                # ), dim=0)

                # 5. Boundary loss
                loss_bound_u = torch.mean(self.loss_function(
                    cond_mask * h * u
                ), dim=1)

                loss_bound_v = torch.mean(self.loss_function(
                    cond_mask * h * v
                ), dim=1)

                loss_bound = loss_bound_u + loss_bound_v

                #
                # Regularizers
                #
                loss_reg = torch.mean(
                    self.loss_function(self.d_dx(h)) + self.loss_function(self.d_dy(h)) + self.loss_function(u) + self.loss_function(v),
                    dim=1
                )

                # Compute combined loss
                loss_tensor = self.params.loss_h * loss_h + self.params.loss_momentum * (loss_u + loss_v) + self.params.loss_bound * loss_bound + self.params.loss_reg * loss_reg

                # Log loss and mean loss
                loss = torch.log(torch.mean(loss_tensor))

                # Compute loss per environment in the batch
                batch_loss = torch.mean(loss_tensor, dim=(1,2))

                # Compute gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Perform an optimization step
                self.optimizer.step()

                # Recycle the data
                self.dataset.tell(h_new, u_new, v_new, batch_loss, random_reset=True)

                # log training metrics
                if i % 10 == 0:
                    loss_tensor = torch.mean(loss_tensor, dim=0).detach().view(self.params.height, self.params.width).cpu().numpy()
                    loss = float(loss.detach().cpu().numpy())
                    loss_h = float(torch.mean(loss_h).detach().cpu().numpy())
                    loss_u = float(torch.mean(loss_u).detach().cpu().numpy())
                    loss_v = float(torch.mean(loss_v).detach().cpu().numpy())
                    # loss_S = float(torch.mean(loss_S).detach().cpu().numpy())
                    # loss_B = float(torch.mean(loss_B).detach().cpu().numpy())
                    loss_bound = float(torch.mean(loss_bound).detach().cpu().numpy())
                    loss_reg = float(torch.mean(loss_reg).detach().cpu().numpy())
                    self.logger.log(f"loss_{self.params.loss}", loss, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_h_{self.params.loss}", loss_h, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_u_{self.params.loss}", loss_u, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_v_{self.params.loss}", loss_v, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_S_{self.params.loss}", loss_S, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_B_{self.params.loss}", loss_B, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_bound_{self.params.loss}", loss_bound, epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log(f"loss_reg_{self.params.loss}", loss_reg, epoch * self.params.n_batches_per_epoch + i)

                    # vRAM stats
                    if torch.cuda.is_available():
                        vram_allocated = torch.cuda.memory_allocated(0)
                    elif torch.backends.mps.is_available():
                        vram_allocated = torch.mps.current_allocated_memory()
                    else: # CPU
                        vram_allocated = 0

                    print(f"Epoch {epoch}, iter.{i}:\tloss: {round(loss,5)};\tloss_bound: {round(loss_bound,5)};\tloss_h: {round(loss_h,5)};\tloss_u: {round(loss_u,5)};\tloss_v: {round(loss_v,5)};\tloss_reg: {round(loss_reg,5)} \tvRAM allocated: {round(vram_allocated/1000000000.0, 2)}GB")

                    #
                    # PLOT LOSS - IF ENABLED
                    #
                    if self.params.plot_loss:
                        loss_tensor -= np.min(loss_tensor)
                        loss_tensor /= np.max(loss_tensor)

                        plot_loss_image.set_data(loss_tensor)

                        plot_loss_total_data = np.append(plot_loss_total_data, np.array([loss]))
                        plot_loss_h_data = np.append(plot_loss_h_data, np.array([self.params.loss_h * loss_h]))
                        plot_loss_momentum_data = np.append(plot_loss_momentum_data, np.array([self.params.loss_momentum * (loss_u + loss_v)]))
                        plot_loss_bound_data = np.append(plot_loss_bound_data, np.array([self.params.loss_bound * loss_bound]))
                        plot_loss_reg_data = np.append(plot_loss_reg_data, np.array([self.params.loss_reg * loss_reg]))

                        plot_loss_total_graph.set_xdata(range(plot_loss_total_data.shape[0]))
                        plot_loss_total_graph.set_ydata(plot_loss_total_data)
                        plot_axs[1].set_xlim([0, plot_loss_total_data.shape[0]])
                        plot_axs[1].set_ylim([np.min(plot_loss_total_data), np.max(plot_loss_total_data)])

                        plot_loss_h_graph.set_xdata(range(plot_loss_h_data.shape[0]))
                        plot_loss_h_graph.set_ydata(plot_loss_h_data)
                        plot_loss_momentum_graph.set_xdata(range(plot_loss_momentum_data.shape[0]))
                        plot_loss_momentum_graph.set_ydata(plot_loss_momentum_data)
                        plot_loss_bound_graph.set_xdata(range(plot_loss_bound_data.shape[0]))
                        plot_loss_bound_graph.set_ydata(plot_loss_bound_data)

                        if self.params.loss_reg > 0:
                            plot_loss_reg_graph.set_xdata(range(plot_loss_reg_data.shape[0]))
                            plot_loss_reg_graph.set_ydata(plot_loss_reg_data)

                        graph_limits = np.concatenate((plot_loss_h_data, plot_loss_momentum_data, plot_loss_bound_data, plot_loss_reg_data))
                        plot_axs[2].set_xlim([0, plot_loss_h_data.shape[0]])
                        plot_axs[2].set_ylim([0, 100])
                        
                if self.params.plot_loss:
                    # Always update the plot to allow interaction
                    # Plot the domain (update existing plot)
                    # Draw updated values
                    plot_fig.canvas.draw()

                    # UI Loop: process all pending UI events
                    plot_fig.canvas.flush_events()

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
        win = Window("Water Layer Thickness", self.params.width, self.params.height)
        win.set_data_range(1.5, 2.5)

        with torch.no_grad():

            # Simulation loop
            while win.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old, cond_mask, flux_x_cond, flux_y_cond = self.dataset.ask()

                # The flow mask is simply 1-cond_mask
                flow_mask = 1 - cond_mask

                # TODO: MAC grid

                # Display water level thickness h
                h = h_old[0, 0].clone()
                # h = h - torch.min(h)
                # h = h / torch.max(h)
                h = h.detach().cpu().numpy()

                win.put_image(h)
                win.update()

                # Predict the new domain state by performing a forward pass through the network
                flux_x, flux_y = self.net(h_old, u_old, v_old, cond_mask, flow_mask, flux_x_cond, flux_y_cond)

                dh_dt = - self.d_dx(flux_x) - self.d_dy(flux_y)

                h_new = h_old + dh_dt * self.params.dt

                h_new = torch.clamp(h_new, min=self.params.Hc)

                u_new = flux_x / h_new
                v_new = flux_y / h_new

                # Store the newly obtained result in the dataset
                self.dataset.tell(h_new, u_new, v_new, random_reset=False)


    def visualize_numerical(self):
        """
        VISUALIZING NUMERICAL REFERENCE SIMULATION
        """

        # Initialize randomization seeds
        torch.manual_seed(1)
        np.random.seed(6)

        # Open a visualization window
        win = Window("Water Layer Thickness", self.params.width, self.params.height)
        win.set_data_range(-1, 1)

        with torch.no_grad():

            # Simulation loop
            while win.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old, cond_mask, flux_x_cond, flux_y_cond = self.dataset.ask()

                # TODO: MAC grid

                # Display water level thickness h
                h = u_old[0, 0].clone()
                print(f"u min max = ({torch.min(u_old)}, {torch.max(u_old)})")
                h = h.detach().cpu().numpy()
                win.put_image(h)
                win.update()

                # Predict the new domain state by numerical simulation
                h = h_old
                u = u_old
                v = v_old
                S = torch.zeros_like(h).to(self.device)

                du_dt = -self.params.grav * self.d_dx(h + S) - u * self.d_dx(u) - v * self.d_dy(u)
                dv_dt = -self.params.grav * self.d_dy(h + S) - u * self.d_dx(v) - v * self.d_dy(v)

                u += du_dt * self.params.dt
                v += dv_dt * self.params.dt

                # Left boundary
                u[:, :, :, 0] = -u[:, :, :, 1]
                v[:, :, :, 0] = v[:, :, :, 1]

                # Right boundary
                u[:, :, :, -1] = -u[:, :, :, -2]
                v[:, :, :, -1] = v[:, :, :, -2]

                # Top
                u[:, :, 0, :] = u[:, :, 1, :]
                v[:, :, 0, :] = -v[:, :, 1, :]

                # Bottom
                u[:, :, -1, :] = u[:, :, -2, :]
                v[:, :, -1, :] = -v[:, :, -2, :]

                dh_dt = - self.d_dx(u * h) - self.d_dy(v * h) # + self.params.Hin

                h += dh_dt * self.params.dt

                h[:, :, :, 0] = h[:, :, :, 1]
                h[:, :, :, -1] = h[:, :, :, -2]
                h[:, :, 0, :] = h[:, :, 1, :]
                h[:, :, -1, :] = h[:, :, -2, :]

                # Store the newly obtained result in the dataset
                self.dataset.tell(h, u, v, random_reset=False)

