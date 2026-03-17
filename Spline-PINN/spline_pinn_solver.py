import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from dataset import Dataset
from loss_calculator import LossCalculator
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

def _dbg_blocking(_desc='',_expr=None):
    _ = _dbg(_desc, _expr)
    _ = input("HIT ENTER TO STEP")
    return _expr

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
        # Loss Calculator module
        #
        self.loss_calculator = LossCalculator(self.dataset, self.params, self.device)

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

        # Training loop:
        # Start from the most recently finished epoch and train until the configured number
        # of epochs has been reached.
        for epoch in range(self.params.load_index, self.params.n_epochs):
            # Each epoch consists of a configurable number of batches.
            for i in range(self.params.n_batches_per_epoch):

                # Ask for a batch from the dataset
                old_hidden_state, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask, grid_offsets, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks, sample_s_conds, sample_s_masks = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                new_hidden_state = self.net(old_hidden_state, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask)

                loss_h, loss_u, loss_v, loss_s, loss_bound = self.loss_calculator.compute_batch_loss(old_hidden_state, new_hidden_state, grid_offsets, h_in, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks, sample_s_conds, sample_s_masks)

                # Log loss (per term)
                if self.params.log_loss:
                    loss_h = torch.log(loss_h + 0.0001) # Add small epsilon to prevent -inf loss due to log
                    loss_u = torch.log(loss_u + 0.0001)
                    loss_v = torch.log(loss_v + 0.0001)
                    loss_s = torch.log(loss_s + 0.0001)
                    loss_bound = torch.log(loss_bound + 0.0001)

                # Compute the loss terms we would like to consider separate learning tasks for PCGrad
                loss_h = torch.mean(loss_h)
                loss_momentum = torch.mean(loss_u + loss_v)
                loss_sediment = torch.mean(loss_s)
                loss_bound = torch.mean(loss_bound)

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
                    self.logger.log("loss_bound", loss_bound.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)

            # Save the training state after each epoch
            if self.params.log:
                self.logger.save_state(self.net, self.optimizer, epoch + 1)

    def eval_mode(self):
        """
        Place the solver in eval mode
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


    def step(self):
        """
        Make one step
        """

        # Ask for a batch from the dataset
        old_hidden_state, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask, grid_offsets, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks, sample_s_conds, sample_s_masks = self.dataset.ask()

        # Predict the new domain state by performing a forward pass through the network
        new_hidden_state = self.net(old_hidden_state, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask)

        # Store the newly obtained result in the dataset
        self.dataset.tell(new_hidden_state)

        # Return the old and new hidden states
        return old_hidden_state, new_hidden_state, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask


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
        window.set_data_range(-0.5, 0.5)

        # Simulation loop
        while window.is_open():

            # Ask for a batch from the dataset
            old_hidden_state, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask, grid_offsets, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks, sample_s_conds, sample_s_masks = self.dataset.ask()

            # Predict the new domain state by performing a forward pass through the network
            new_hidden_state = self.net(old_hidden_state, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask)

            # loss_h, loss_u, loss_v, loss_bound, loss_damp = self.compute_batch_loss(old_hidden_state, new_hidden_state, grid_offsets, sample_h_conds, sample_h_masks, sample_uv_conds, sample_uv_masks)

            # Interpolate spline coefficients to obtain the necessary quantities
            h, grad_h, hu, grad_hu, hv, grad_hv, s, grad_s = self.dataset.interpolate_superres(new_hidden_state, self.params.resolution_factor)

            # Store the newly obtained result in the dataset
            self.dataset.tell(new_hidden_state)

            # Display water level thickness h
            h = h[0, 0].clone()
            # h = h - torch.min(h)
            # h = h / torch.max(h)
            h = h.detach().cpu().numpy()

            window.put_image(h)
            window.update()