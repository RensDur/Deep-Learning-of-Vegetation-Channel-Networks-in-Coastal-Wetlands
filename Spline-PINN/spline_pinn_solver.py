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
        self.net = get_Net(params, self.dataset.hidden_size).to(self.device)

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

        # Training loop:
        # Start from the most recently finished epoch and train until the configured number
        # of epochs has been reached.
        for epoch in range(self.params.load_index, self.params.n_epochs):
            # Each epoch consists of a configurable number of batches.
            for i in range(self.params.n_batches_per_epoch):

                # Ask for a batch from the dataset
                old_hidden_state, u_cond, u_mask, v_cond, v_mask, grid_offsets, sample_u_conds, sample_u_masks, sample_v_conds, sample_v_masks = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                new_hidden_state = self.net(old_hidden_state, u_cond, u_mask, v_cond, v_mask)


                # Compute Physics Informed Loss
                loss_total = 0

                # Go over each sample
                for j, sample in enumerate(grid_offsets):
                    offset = torch.floor(sample*self.params.resolution_factor)/self.params.resolution_factor

                    # For added clarity: The masks define where the BCs act, they're 1 everywhere on the boundary, 0 everywhere else
                    sample_u_cond = sample_u_conds[j]
                    sample_u_mask = sample_u_masks[j]
                    sample_v_cond = sample_v_conds[j]
                    sample_v_mask = sample_v_masks[j]

                    sample_u_domain_mask = 1-sample_u_mask
                    sample_v_domain_mask = 1-sample_v_mask

                    # Put additional border_weight on domain boundaries:
                    # Important: weighed by parameter 'border_weight'
                    sample_u_mask = (sample_u_mask + sample_u_mask*self.diffuse(sample_u_domain_mask)*self.params.border_weight).detach()
                    sample_v_mask = (sample_v_mask + sample_v_mask*self.diffuse(sample_v_domain_mask)*self.params.border_weight).detach()

                    # Interpolate spline coefficients to obtain the necessary quantities
                    h, grad_h, dh_dt, u, grad_u, laplace_u, du_dt, v, grad_v, laplace_v, dv_dt = self.dataset.interpolate_states(old_hidden_state, new_hidden_state, offset)

                    grad_uh = h * grad_u + u * grad_h
                    grad_vh = h * grad_v + v * grad_h

                    #
                    # COMPUTE SAMPLE LOSS
                    #

                    # TODO: What is the shape of grad...? (detach dx and dy)

                    # h-loss
                    loss_h = torch.mean(self.loss_function(
                        dh_dt + grad_uh + grad_vh # - self.params.Hin
                    ), dim=1)

                    # Momentum loss
                    loss_u = du_dt + self.params.grav * 






                




                # Compute combined loss
                loss = torch.mean(torch.log(
                    self.params.loss_h * loss_h
                    + self.params.loss_momentum * (loss_u + loss_v)
                    # + self.params.loss_S * loss_S
                    # + self.params.loss_B * loss_B
                    + self.params.loss_bound * loss_bound
                    + self.params.loss_reg * loss_reg
                ))

                # Compute gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Perform an optimization step
                self.optimizer.step()

                # Recycle the data
                self.dataset.tell(h_new, u_new, v_new, random_reset=True)

                # log training metrics
                if i % 10 == 0:
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

                    print(f"Epoch {epoch}, iter.{i}:\tloss: {round(loss,5)};\tloss_bound: {round(loss_bound,5)};\tloss_h: {round(loss_h,5)};\tloss_u: {round(loss_u,5)};\tloss_v: {round(loss_v,5)};\tloss_reg: {round(loss_reg,5)} \tvRAM allocated: {round(torch.mps.current_allocated_memory()/1000000000.0, 2)}GB")
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
        win = Window("Water Layer Thickness", self.params.width, self.params.height)
        win.set_data_range(self.params.H0 - 0.0005, self.params.H0+0.0005)
        win.set_data_range(self.params.H0 - 1e-2, self.params.H0 + 1e-2)

        with torch.no_grad():

            # Simulation loop
            while win.is_open():

                # Ask for a batch from the dataset
                h_old, u_old, v_old = self.dataset.ask()

                # TODO: MAC grid

                # Display water level thickness h
                h = h_old[0, 0].clone()
                # h = h - torch.min(h)
                # h = h / torch.max(h)
                h = h.detach().cpu().numpy()

                win.put_image(h)
                win.update()

                # Predict the new domain state by performing a forward pass through the network
                h_new, u_new, v_new = self.net(h_old, u_old, v_old)

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

