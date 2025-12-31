import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from dataset import Dataset
from spline_models import get_Net
import parameters
from Logger import Logger,t_step
import cv2
import time
from window import Window
import matplotlib.pyplot as plt


class SplinePINNSolver:
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
        self.net = get_Net(params, self.dataset.hidden_size()).to(self.device)

        #
        # Diffusion operation (needed, if we want to put more loss-weight to regions close to the domain boundaries)
        #
        self.kernel_width = 3
        self.kernel = torch.exp(-torch.arange(-2,2.001,4/(2*self.kernel_width)).float()**2)
        self.kernel /= torch.sum(self.kernel)
        self.kernel_x = self.kernel.unsqueeze(0).unsqueeze(1).unsqueeze(3).to(self.device)
        self.kernel_y = self.kernel.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(self.device)

    def diffuse(self, T):
        """
        Needed to put extra weight on domain borders
        """
        T = F.conv2d(T,self.kernel_x,padding=[self.kernel_width,0])
        T = F.conv2d(T,self.kernel_y,padding=[0,self.kernel_width])
        return T

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

            # Perform warmup if requested
            if self.params.n_warmup_steps is not None:
                self.net.eval()
                for i in range(self.params.n_warmup_steps):
                    z_cond,z_mask,old_hidden_state,_,_,_ = self.dataset.ask()
                    new_hidden_state = self.net(old_hidden_state,z_cond,z_mask)
                    self.dataset.tell(new_hidden_state)
                    if i%(self.params.n_warmup_steps//100)==0:
                        print(f"warmup {i/(self.params.n_warmup_steps//100)} %")
        self.params.load_index = 0 if self.params.load_index is None else self.params.load_index

        # Enable training of the model
        self.net.train()

        #
        # Prepare Loss Plots
        #
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
            # plot_loss_reg_data = np.array([])

            plot_loss_total_graph = plot_axs[1].plot(range(plot_loss_total_data.shape[0]), plot_loss_total_data)[0]

            plot_loss_h_graph = plot_axs[2].plot(range(plot_loss_h_data.shape[0]), plot_loss_h_data, label="h-loss")[0]
            plot_loss_momentum_graph = plot_axs[2].plot(range(plot_loss_momentum_data.shape[0]), plot_loss_momentum_data, label="u,v-loss")[0]
            plot_loss_bound_graph = plot_axs[2].plot(range(plot_loss_bound_data.shape[0]), plot_loss_bound_data, label="bound-loss")[0]
            # plot_loss_reg_graph = plot_axs[2].plot(range(plot_loss_reg_data.shape[0]), plot_loss_reg_data, label="reg-loss")[0]

            plot_axs[2].legend(handles=[plot_loss_h_graph, plot_loss_momentum_graph, plot_loss_bound_graph], loc="upper right")

            plt.show()


        # Training loop:
        # Start from the most recently finished epoch and train until the configured number
        # of epochs has been reached.
        for epoch in range(self.params.load_index, self.params.n_epochs):
            # Each epoch consists of a configurable number of batches.
            for i in range(self.params.n_batches_per_epoch):

                # Ask for a batch from the dataset
                old_hidden_state, h_cond, h_mask, uv_cond, uv_mask, grid_offsets, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                new_hidden_state = self.net(old_hidden_state, h_cond, h_mask, uv_cond, uv_mask)

                # Compute Physics Informed Loss image tensor
                loss_tensor = 0

                # Go over each sample
                for j, sample in enumerate(grid_offsets):
                    offset = torch.floor(sample*self.params.resolution_factor)/self.params.resolution_factor

                    # For added clarity: The masks define where the BCs act, they're 1 everywhere on the boundary, 0 everywhere else
                    sample_h_cond = sample_h_conds[j]
                    sample_h_mask = sample_h_masks[j]
                    sample_uv_cond = sample_uv_conds[j]
                    sample_uv_mask = sample_uv_masks[j]

                    sample_h_domain_mask = 1-sample_h_mask
                    sample_uv_domain_mask = 1-sample_uv_mask

                    # Put additional border_weight on domain boundaries:
                    # Important: weighed by parameter 'border_weight'
                    sample_h_mask = (sample_h_mask + sample_h_mask*self.diffuse(sample_h_domain_mask)*self.params.border_weight).detach()
                    sample_uv_mask = (sample_uv_mask + sample_uv_mask*self.diffuse(sample_uv_domain_mask)*self.params.border_weight).detach()

                    # Interpolate spline coefficients to obtain the necessary quantities
                    h, grad_h, dh_dt, u, grad_u, laplace_u, du_dt, v, grad_v, laplace_v, dv_dt = self.dataset.interpolate_states(old_hidden_state, new_hidden_state, offset)

                    grad_uh = h * grad_u + u * grad_h
                    grad_vh = h * grad_v + v * grad_h

                    #
                    # COMPUTE SAMPLE LOSS
                    #

                    # Boundary loss
                    loss_bound_h = torch.mean(
                        self.loss_function(sample_h_mask[:,:,1:-1,1:-1] * (h - sample_h_cond[:,:,1:-1,1:-1]))
                        ,dim=1
                    )

                    loss_bound_grad_h = torch.mean(
                        self.loss_function(sample_uv_mask[:,:,1:-1,1:-1] * grad_h)
                        ,dim=1
                    )

                    loss_bound_u = torch.mean(
                        self.loss_function(sample_uv_mask[:,:,1:-1,1:-1] * (u - sample_uv_cond[:,:,1:-1,1:-1]))
                        ,dim=1
                    )

                    loss_bound_v = torch.mean(
                        self.loss_function(sample_uv_mask[:,:,1:-1,1:-1] * (v - sample_uv_cond[:,:,1:-1,1:-1]))
                        ,dim=1
                    )

                    loss_bound = loss_bound_h + loss_bound_grad_h + loss_bound_u + loss_bound_v

                    # h-loss
                    loss_h = torch.mean(self.loss_function(
                        dh_dt + grad_uh[:,0:1] + grad_vh[:,1:2] # - self.params.Hin
                    ), dim=1)

                    # Momentum loss
                    loss_u = du_dt + self.params.grav * grad_h[:,0:1] + u * grad_u[:,0:1] + v * grad_u[:,1:2]
                    loss_v = dv_dt + self.params.grav * grad_h[:,1:2] + u * grad_v[:,0:1] + v * grad_v[:,1:2]

                    loss_u = torch.mean(self.loss_function(loss_u), dim=1)
                    loss_v = torch.mean(self.loss_function(loss_v), dim=1)

                    loss_tensor += self.params.loss_bound * loss_bound + self.params.loss_h * loss_h + self.params.loss_momentum * (loss_u + loss_v)

                # Normalize towards the number of samples taken
                loss_tensor /= self.params.n_samples

                # Aside from the loss image, also compute mean loss
                loss_total = torch.mean(loss_tensor)

                # If configured, compute log loss
                if self.params.log_loss:
                    loss_total = torch.log(loss_total)

                # Reset old gradients to 0 and compute new gradients with backpropagation
                self.net.zero_grad()
                loss_total.backward()

                # Clip gradients
                if self.params.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(),self.params.clip_grad_value)

                if self.params.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),self.params.clip_grad_norm)
                
                # Perform an optimization step
                self.optimizer.step()

                # Recycle the data
                self.dataset.tell(new_hidden_state)

                #
                # Plotting and logging
                #

                if i % 10 == 0:
                    

                    # self.logger.log(f"loss_total", loss_total, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_h", loss_h, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_u", loss_u, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_v", loss_v, epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log(f"loss_bound", loss_bound, epoch * self.params.n_batches_per_epoch + i)

                    # # vRAM stats
                    # if torch.cuda.is_available():
                    #     vram_allocated = torch.cuda.memory_allocated(0)
                    # elif torch.backends.mps.is_available():
                    #     vram_allocated = torch.mps.current_allocated_memory()
                    # else: # CPU
                    #     vram_allocated = 0

                    # print(f"Epoch {epoch}, iter.{i}:\tloss: {round(loss_total,5)};\tloss_bound: {round(loss_bound,5)};\tloss_h: {round(loss_h,5)};\tloss_u: {round(loss_u,5)};\tloss_v: {round(loss_v,5)};\tvRAM allocated: {round(vram_allocated/1000000000.0, 2)}GB")

                    print(f"Epoch {epoch}/{self.params.n_epochs}, iteration {i}")

                    #
                    # PLOT LOSS - IF ENABLED
                    #
                    if self.params.plot_loss:
                        loss_tensor = torch.mean(loss_tensor, dim=0).detach().view(self.params.height-2, self.params.width-2).cpu().numpy()
                        loss_total = float(loss_total.detach().cpu().numpy())
                        loss_h = float(torch.mean(loss_h).detach().cpu().numpy())
                        loss_u = float(torch.mean(loss_u).detach().cpu().numpy())
                        loss_v = float(torch.mean(loss_v).detach().cpu().numpy())
                        loss_bound = float(torch.mean(loss_bound).detach().cpu().numpy())

                        loss_tensor -= np.min(loss_tensor)
                        loss_tensor /= np.max(loss_tensor)

                        plot_loss_image.set_data(loss_tensor)

                        plot_loss_total_data = np.append(plot_loss_total_data, np.array([loss_total]))
                        plot_loss_h_data = np.append(plot_loss_h_data, np.array([self.params.loss_h * loss_h]))
                        plot_loss_momentum_data = np.append(plot_loss_momentum_data, np.array([self.params.loss_momentum * (loss_u + loss_v)]))
                        plot_loss_bound_data = np.append(plot_loss_bound_data, np.array([self.params.loss_bound * loss_bound]))

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

                        graph_limits = np.concatenate((plot_loss_h_data, plot_loss_momentum_data, plot_loss_bound_data))
                        plot_axs[2].set_xlim([0, plot_loss_h_data.shape[0]])
                        plot_axs[2].set_ylim([0, 500])

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
        win = Window("Water Layer Thickness", self.params.width * self.params.resolution_factor, self.params.height * self.params.resolution_factor)
        win.set_data_range(1, 3)

        # Simulation loop
        while win.is_open():

            # Ask for a batch from the dataset
            old_hidden_state, h_cond, h_mask, _, _, _ = self.dataset.ask()

            # Predict the new domain state by performing a forward pass through the network
            new_hidden_state = self.net(old_hidden_state, h_cond, h_mask)

            # Interpolate spline coefficients to obtain the necessary quantities
            h, grad_h, u, grad_u, laplace_u, v, grad_v, laplace_v = self.dataset.interpolate_superres(new_hidden_state, self.params.resolution_factor)

            # Store the newly obtained result in the dataset
            self.dataset.tell(new_hidden_state)

            # Display water level thickness h
            h = h[0, 0].clone()
            # h = h - torch.min(h)
            # h = h / torch.max(h)
            h = h.detach().cpu().numpy()

            win.put_image(h)
            win.update()


    def visualize_numerical(self):
        """
        VISUALIZING NUMERICAL REFERENCE SIMULATION
        """

        # Initialize randomization seeds
        torch.manual_seed(1)
        np.random.seed(6)

        # Open a visualization window
        win = Window("Water Layer Thickness", self.params.width, self.params.height)
        win.set_data_range(self.params.H0 - 0.0005, self.params.H0+0.0005)

        with torch.no_grad():

            # Simulation loop
            while win.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old = self.dataset.ask()

                # TODO: MAC grid

                # Display water level thickness h
                h = h_old[0, 0].clone()
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

