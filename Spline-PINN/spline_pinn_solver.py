import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from dataset import Dataset
from spline_models import get_Net
import parameters
from Logger import Logger
import matplotlib.pyplot as plt
from pcgrad.pcgrad import PCGrad
import os
import psutil

def _dbg(_desc='',_expr=None):
    print(f"DBG! >> {_desc}: {_expr}")
    return _expr

def _dbg_nan(_desc='',_tensor=torch.zeros(1)):
    _expr = torch.mean(_tensor).detach().cpu()
    _dbg(_desc, _expr)
    return _tensor

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
        self.net = get_Net(params, self.dataset.variables).to(self.device)

        #
        # Diffusion operation (needed, if we want to put more loss-weight to regions close to the domain boundaries)
        #
        self.kernel_width = 3
        self.kernel = torch.exp(-torch.arange(-2,2.001,4/(2*self.kernel_width)).float()**2)
        self.kernel /= torch.sum(self.kernel)
        self.kernel_x = self.kernel.unsqueeze(0).unsqueeze(1).unsqueeze(3).to(self.device)
        self.kernel_y = self.kernel.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(self.device)

        self.damp_loss_factor = 1000

    def diffuse(self, T):
        """
        Needed to put extra weight on domain borders
        """
        T = F.conv2d(T,self.kernel_x,padding=[self.kernel_width,0])
        T = F.conv2d(T,self.kernel_y,padding=[0,self.kernel_width])
        return T

    def loss_function(self, x):
        # return F.huber_loss(x, torch.zeros_like(x), reduction="none", delta=self.params.huber_delta)
        return x**2
    
    def compute_batch_loss(self, old_hidden_state, new_hidden_state, grid_offsets, sample_grad_h_conds, sample_grad_h_masks, sample_u_conds, sample_u_masks, sample_v_conds, sample_v_masks, sample_S_conds, sample_S_masks, sample_grad_S_conds, sample_grad_S_masks, dim=[1,2,3]):

        # Compute Physics Informed Loss image tensor
        loss_h = 0
        loss_u = 0
        loss_v = 0
        loss_S = 0
        loss_B = 0
        loss_bound = 0

        # Go over each sample
        for j, sample in enumerate(grid_offsets):
            offset = torch.floor(sample*self.params.resolution_factor)/self.params.resolution_factor

            # For added clarity: The masks define where the BCs act, they're 1 everywhere on the boundary, 0 everywhere else
            sample_grad_h_cond = sample_grad_h_conds[j]
            sample_grad_h_mask = sample_grad_h_masks[j]
            sample_u_cond = sample_u_conds[j]
            sample_u_mask = sample_u_masks[j]
            sample_v_cond = sample_v_conds[j]
            sample_v_mask = sample_v_masks[j]
            sample_S_cond = sample_S_conds[j]
            sample_S_mask = sample_S_masks[j]
            sample_grad_S_cond = sample_grad_S_conds[j]
            sample_grad_S_mask = sample_grad_S_masks[j]

            sample_grad_h_domain_mask = 1-sample_grad_h_mask
            sample_u_domain_mask = 1-sample_u_mask
            sample_v_domain_mask = 1-sample_v_mask
            sample_S_domain_mask = 1-sample_S_mask
            sample_grad_S_domain_mask = 1-sample_grad_S_mask

            # Put additional border_weight on domain boundaries:
            # Important: weighed by parameter 'border_weight'
            # sample_grad_h_mask = (sample_grad_h_mask + sample_grad_h_mask*self.diffuse(sample_grad_h_domain_mask)*self.params.border_weight).detach()
            # sample_u_mask = (sample_u_mask + sample_u_mask*self.diffuse(sample_u_domain_mask)*self.params.border_weight).detach()
            # sample_v_mask = (sample_v_mask + sample_v_mask*self.diffuse(sample_v_domain_mask)*self.params.border_weight).detach()
            # sample_S_mask = (sample_S_mask + sample_S_mask*self.diffuse(sample_S_domain_mask)*self.params.border_weight).detach()
            # sample_grad_S_mask = (sample_grad_S_mask + sample_grad_S_mask*self.diffuse(sample_grad_S_domain_mask)*self.params.border_weight).detach()

            # Interpolate spline coefficients to obtain the necessary quantities
            h, grad_h, dh_dt, u, grad_u, laplace_u, du_dt, v, grad_v, laplace_v, dv_dt, S, grad_S, laplace_S, dS_dt = self.dataset.interpolate_states(old_hidden_state, new_hidden_state, offset)

            # Temporarily replace B fields with zero
            B = torch.zeros_like(S)
            grad_B = torch.zeros_like(grad_S)
            laplace_B = torch.zeros_like(laplace_S)

            #
            # COMPUTE SAMPLE LOSS
            #


            # Compute intermediate terms
            h_total = h + self.params.H0

            # water levels can never reach negative levels, in those locations where it does, the hydrodynamics laws will not hold anymore
            # therefore, we cannot evaluate the environment on how well it adheres to those laws in those places.
            # We make a mask that's 1 everywhere there's enough water present to evaluate the system on its performance there.
            h_total = F.relu(h_total - self.params.Hc) + self.params.Hc

            bed_roughness_coefficient_n = self.params.nb + (self.params.nv - self.params.nb) * (B / self.params.k)
            chezy_coefficient = (1.0 / bed_roughness_coefficient_n) * torch.pow(h_total, 1.0/6.0)
            tau_bx_per_rho = (self.params.grav / torch.pow(chezy_coefficient, 2.0)) * torch.pow(u**2 + v**2 + 0.001, 0.5) * u
            tau_by_per_rho = (self.params.grav / torch.pow(chezy_coefficient, 2.0)) * torch.pow(u**2 + v**2 + 0.001, 0.5) * v
            tau_b_per_rho = (self.params.grav / torch.pow(chezy_coefficient, 2.0)) * (u**2 + v**2)

            he = h_total - self.params.Hc

            # Compute topographic diffusion term (\/(Ds\/S)) (Ds is a field)
            Ds = self.params.D0 * (1.0 - self.params.pD * (B / self.params.k))
            grad_Ds = - ((self.params.D0 * self.params.pD) / self.params.k) * grad_B

            gradDs_dot_gradS = grad_Ds[:, 1:2] * grad_S[:, 1:2] + grad_Ds[:, 0:1] * grad_S[:, 0:1]

            # Full divergence term
            div_Ds_grad_S = Ds * laplace_S + gradDs_dot_gradS

            # h-loss
            loss_h = loss_h + torch.mean(self.loss_function(
                dh_dt + (grad_u[:,1:2] + grad_v[:,0:1]) * (h + self.params.H0) + (grad_h[:,1:2]*u + grad_h[:,0:1]*v) + self.params.epsilon * h
            ), dim)

            # Momentum loss
            loss_u = loss_u + torch.mean(self.loss_function(
                du_dt - self.params.Du * laplace_u + self.params.grav * (grad_h[:,1:2] + grad_S[:,1:2]) + self.params.k_epsilon*u + u * grad_u[:,1:2] + v * grad_u[:,0:1] - self.params.f_epsilon * v + tau_bx_per_rho/h_total
            ), dim)

            loss_v = loss_v + torch.mean(self.loss_function(
                dv_dt - self.params.Du * laplace_v + self.params.grav * (grad_h[:,0:1] + grad_S[:,0:1]) + self.params.k_epsilon*v + u * grad_v[:,1:2] + v * grad_v[:,0:1] + self.params.f_epsilon * u + tau_by_per_rho/h_total
            ), dim)

            loss_S = loss_S + torch.mean(self.loss_function(
                dS_dt - 1.0 * (self.params.Sin * (he / (self.params.Qs + he)) - self.params.Es * (1.0 - self.params.pE * (B / self.params.k)) * S * tau_b_per_rho + div_Ds_grad_S)
            ), dim)

            # loss_B = loss_B + torch.mean(self.loss_function(
            #     dB_dt - 1.0 * (self.params.r * B * (1.0 - (B / self.params.k)) * (self.params.Qq / (self.params.Qq + he)) - self.params.EB * B * tau_b_per_rho + self.params.DB * laplace_B)
            # ), dim)

            #
            # Boundary loss
            #

            loss_bound_grad_h = torch.mean(sample_grad_h_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_h - sample_grad_h_cond[:,:,1:-1,1:-1]
            ), dim)

            loss_bound_u = torch.mean(sample_u_mask[:,:,1:-1,1:-1] * self.loss_function(
                u - sample_u_cond[:,:,1:-1,1:-1]
            ), dim)

            loss_bound_v = torch.mean(sample_v_mask[:,:,1:-1,1:-1] * self.loss_function(
                v - sample_v_cond[:,:,1:-1,1:-1]
            ), dim)

            loss_bound_S = torch.mean(sample_S_mask[:,:,1:-1,1:-1] * self.loss_function(
                S - sample_S_cond[:,:,1:-1,1:-1]
            ), dim)

            loss_bound_grad_S = torch.mean(sample_grad_S_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_S - sample_grad_S_cond[:,:,1:-1,1:-1]
            ), dim)

            # Experiment condition: S may never become negative
            loss_s_negative = torch.mean(self.loss_function(
                torch.relu(-S) # Whenever S reaches below-zero values, they're flipped and ReLU-d so the network is penalized by negative values for S
            ), dim)

            loss_bound = loss_bound + (loss_bound_grad_h + loss_bound_u + loss_bound_v + loss_bound_S + loss_bound_grad_S)

        # Multiply by the loss weights
        loss_h = loss_h * self.params.loss_h
        loss_u = loss_u * self.params.loss_momentum
        loss_v = loss_v * self.params.loss_momentum
        loss_S = loss_S * self.params.loss_sediment * self.params.morphological_acc_factor
        # loss_B = loss_B * self.params.loss_vegetation * self.params.morphological_acc_factor
        loss_bound = loss_bound * self.params.loss_bound

        # Normalize towards the number of samples taken
        loss_h = loss_h / self.params.n_samples
        loss_u = loss_u / self.params.n_samples
        loss_v = loss_v / self.params.n_samples
        loss_S = loss_S / self.params.n_samples
        # loss_B = loss_B / self.params.n_samples
        loss_bound = loss_bound / self.params.n_samples

        return loss_h, loss_u, loss_v, loss_S, loss_bound

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
        self.optimizer = PCGrad(self.optimizer)

        torch.autograd.set_detect_anomaly(True)

        #
        # Logger
        #
        self.logger = Logger(parameters.get_description(self.params), use_csv=self.params.log_csv, use_tensorboard=self.params.log_tensorboard)
        if self.params.load_latest or self.params.load_date_time is not None or self.params.load_index is not None:
            self.load_logger = Logger(parameters.get_description(self.params), use_csv=self.params.log_csv, use_tensorboard=self.params.log_tensorboard)
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
                    h_cond,h_mask,old_hidden_state,_,_,_ = self.dataset.ask()
                    new_hidden_state = self.net(old_hidden_state,h_cond,h_mask)
                    self.dataset.tell(new_hidden_state)
                    if i%(self.params.n_warmup_steps//100)==0:
                        print(f"warmup {i/(self.params.n_warmup_steps//100)} %")
        self.params.load_index = 0 if self.params.load_index is None else self.params.load_index

        # Write experiment info to info.txt
        info_text = "Parameters:\n\n"
        for key, value in vars(self.params).items():
            info_text += f"{key}: {value}\n"
        self.logger.log_info(info_text)

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
            plot_loss_image = plot_axs[0].imshow(np.zeros((self.params.height-2, self.params.width-2)), cmap="gray", vmin=0, vmax=1)

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
                old_hidden_state, grad_h_cond, grad_h_mask, u_cond, u_mask, v_cond, v_mask, S_cond, S_mask, grad_S_cond, grad_S_mask, grid_offsets, sample_grad_h_conds, sample_grad_h_masks, sample_u_conds, sample_u_masks, sample_v_conds, sample_v_masks, sample_S_conds, sample_S_masks, sample_grad_S_conds, sample_grad_S_masks = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                new_hidden_state = self.net(old_hidden_state, grad_h_cond, grad_h_mask, u_cond, u_mask, v_cond, v_mask, S_cond, S_mask, grad_S_cond, grad_S_mask)

                dim = [1,2,3]
                if self.params.plot_loss:
                    dim = [1]

                loss_h, loss_u, loss_v, loss_S, loss_bound = self.compute_batch_loss(old_hidden_state, new_hidden_state, grid_offsets, sample_grad_h_conds, sample_grad_h_masks, sample_u_conds, sample_u_masks, sample_v_conds, sample_v_masks, sample_S_conds, sample_S_masks, sample_grad_S_conds, sample_grad_S_masks, dim)

                if self.params.plot_loss:
                    # Combine the losses to create a loss_tensor image
                    loss_tensor = torch.mean(loss_h + loss_u + loss_v + loss_S + loss_bound, dim=0)

                    # Compute total loss value
                    loss_total = torch.log(torch.mean(loss_tensor))

                    # Restore correct averaging of individual loss components
                    loss_h = torch.mean(loss_h, dim=[1,2])
                    loss_u = torch.mean(loss_u, dim=[1,2])
                    loss_v = torch.mean(loss_v, dim=[1,2])
                    loss_S = torch.mean(loss_S, dim=[1,2])
                    # loss_B = torch.mean(loss_B, dim=[1,2])
                    loss_bound = torch.mean(loss_bound, dim=[1,2])

                # Compute the loss terms we would like to consider separate learning tasks for PCGrad
                loss_h = torch.mean(loss_h)
                loss_momentum = torch.mean(loss_u + loss_v)
                loss_sediment = torch.mean(loss_S)
                # loss_vegetation = torch.mean(loss_B)
                loss_bound = torch.mean(loss_bound)

                # TODO: Temporary test: less loss objectives
                loss_objective_hydrodynamics = loss_h + loss_momentum
                loss_objective_sediment_vegetation = loss_sediment# + loss_vegetation

                # Log loss (per term)
                if self.params.log_loss:
                    loss_h = torch.log(loss_h + 0.0001)
                    loss_momentum = torch.log(loss_momentum + 0.0001)
                    loss_sediment = torch.log(loss_sediment + 0.0001)
                    # loss_vegetation = torch.log(loss_vegetation + 0.0001)
                    loss_bound = torch.log(loss_bound + 0.0001)

                    loss_objective_hydrodynamics = torch.log(loss_objective_hydrodynamics + 0.0001)
                    loss_objective_sediment_vegetation = torch.log(loss_objective_sediment_vegetation + 0.0001)

                # For backprop using PCGrad, construct each loss term
                pcgrad_losses = [
                    loss_h,
                    loss_momentum,
                    loss_sediment,
                    loss_bound
                ]

                # Reset old gradients to 0 and compute new gradients with backpropagation
                self.net.zero_grad()
                
                # PCGrad backprop pass
                self.optimizer.pc_backward(pcgrad_losses)

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

                    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB
                    ram_usage = round(ram_usage, 2)

                    max_vram_allocated = None
                    max_vram_reserved = None

                    if torch.cuda.is_available():
                        max_vram_allocated = torch.cuda.memory.max_memory_allocated() / (1024 * 1024)
                        max_vram_reserved = torch.cuda.memory.max_memory_reserved() / (1024 * 1024)

                        max_vram_allocated = round(max_vram_allocated, 2)
                        max_vram_reserved = round(max_vram_reserved, 2)

                    print(f"Epoch {epoch}/{self.params.n_epochs}, iteration {i} \t RAM: {ram_usage}MB \t vRAM: {max_vram_allocated}/{max_vram_reserved} (MAX. allocated/reserved, MB)")

                    # Log the loss to csv and tensorboard
                    self.logger.log("loss_h", loss_h.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_momentum", loss_momentum.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_sediment", loss_sediment.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    # self.logger.log("loss_vegetation", loss_vegetation.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_bound", loss_bound.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)

                    self.logger.log("loss_objective_hydrodynamics", loss_objective_hydrodynamics.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_objective_sediment_vegetation", loss_objective_sediment_vegetation.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)

                    # log_index = epoch * self.params.n_batches_per_epoch + i
                    # self.logger.log_all(["loss_h", "loss_momentum", "loss_bound"], [loss_h.detach().cpu(), loss_momentum.detach().cpu(), loss_bound.detach().cpu()], log_index)

                    #
                    # PLOT LOSS - IF ENABLED
                    #
                    if self.params.plot_loss:
                        loss_total = float(loss_total.detach().cpu().numpy())
                        loss_tensor = loss_tensor.detach().cpu().numpy()
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
                        plot_axs[2].set_ylim([np.min(graph_limits), np.max(graph_limits)])

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




    def visualize(self, window):
        """
        VISUALIZING RESULTS
        """

        # Initialize randomization seeds
        torch.manual_seed(1)
        np.random.seed(6)

        #
        # Logger
        #
        self.logger = Logger(parameters.get_description(self.params), use_csv=False, use_tensorboard=False, device=self.device)

        # Load the trained model state
        date_time, index = self.logger.load_state(self.net, None, datetime=self.params.load_date_time, index=self.params.load_index)

        # Enable evaluation of the model
        self.net.eval()

        print(f"Loaded {self.params.net}: {date_time}, index: {index}")

        # Open a visualization window
        window.set_data_range(-1, 1)

        # Simulation loop
        while window.is_open():

            # Ask for a batch from the dataset
            old_hidden_state, grad_h_cond, grad_h_mask, u_cond, u_mask, v_cond, v_mask, S_cond, S_mask, grad_S_cond, grad_S_mask, grid_offsets, sample_grad_h_conds, sample_grad_h_masks, sample_u_conds, sample_u_masks, sample_v_conds, sample_v_masks, sample_S_conds, sample_S_masks, sample_grad_S_conds, sample_grad_S_masks = self.dataset.ask()

            # Predict the new domain state by performing a forward pass through the network
            new_hidden_state = self.net(old_hidden_state, grad_h_cond, grad_h_mask, u_cond, u_mask, v_cond, v_mask, S_cond, S_mask, grad_S_cond, grad_S_mask)

            # loss_h, loss_u, loss_v, loss_bound, loss_damp = self.compute_batch_loss(old_hidden_state, new_hidden_state, grid_offsets, sample_uv_conds, sample_uv_masks, sample_h_conds, sample_h_masks, sample_S_conds, sample_S_masks)

            # Interpolate spline coefficients to obtain the necessary quantities
            h, grad_h, u, grad_u, laplace_u, v, grad_v, laplace_v, S, grad_S, laplace_S = self.dataset.interpolate_superres(new_hidden_state, self.params.resolution_factor)

            # Store the newly obtained result in the dataset
            self.dataset.tell(new_hidden_state)

            # Display water level thickness h
            # h = grad_h_mask[0, 0].clone()
            # h = u_mask[0, 0].clone()
            # h = v_mask[0, 0].clone()
            # h = S_mask[0, 0].clone()
            # h = grad_S_mask[0, 1].clone()

            h = u_cond[0, 0].clone()

            # h = h - torch.min(h)
            # h = h / torch.max(h)
            h = h.detach().cpu().numpy()

            window.put_image(h)
            window.update()


    def visualize_numerical(self, window):
        """
        VISUALIZING NUMERICAL REFERENCE SIMULATION
        """

        # Initialize randomization seeds
        torch.manual_seed(1)
        np.random.seed(6)

        # Open a visualization window
        window.set_data_range(self.params.H0 - 0.0005, self.params.H0+0.0005)

        with torch.no_grad():

            # Simulation loop
            while window.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old = self.dataset.ask()

                # TODO: MAC grid

                # Display water level thickness h
                h = h_old[0, 0].clone()
                h = h.detach().cpu().numpy()
                window.put_image(h)
                window.update()

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

