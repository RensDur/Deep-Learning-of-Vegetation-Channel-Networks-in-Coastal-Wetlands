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
import threading
import time

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
        self.nets = get_Net(params, self.dataset.variables)
        self.water_net = self.nets[0].to(self.device)
        self.sediment_net = self.nets[1].to(self.device)
        self.vegetation_net = self.nets[2].to(self.device)

        #
        # Training Stage
        #
        self.training_sediment = False
        self.training_sediment_start_epoch = 1000

        self.training_vegetation = False
        self.training_vegetation_start_epoch = 1000

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
        return torch.pow(x, 2)
    
    def compute_batch_loss(self, old_hidden_state, new_hidden_state, grid_offsets, sample_closed_masks, sample_opened_masks, sample_h_masks, sample_h_conds, dim=[1,2,3]):


        # Stack all the masks in batch dimension
        sample_closed_mask = torch.cat([sample_closed_masks[j] for j in range(len(sample_closed_masks))], dim=0)
        sample_opened_mask = torch.cat([sample_opened_masks[j] for j in range(len(sample_opened_masks))], dim=0)
        sample_h_mask = torch.cat([sample_h_masks[j] for j in range(len(sample_h_masks))], dim=0)
        sample_h_cond = torch.cat([sample_h_conds[j] for j in range(len(sample_h_conds))], dim=0)

        all_samples__h = []
        all_samples__grad_h = []
        all_samples__dh_dt = []
        all_samples__u = []
        all_samples__grad_u = []
        all_samples__laplacian_u = []
        all_samples__du_dt = []
        all_samples__v = []
        all_samples__grad_v = []
        all_samples__laplacian_v = []
        all_samples__dv_dt = []
        all_samples__s = []
        all_samples__grad_s = []
        all_samples__laplacian_s = []
        all_samples__ds_dt = []
        all_samples__b = []
        all_samples__grad_b = []
        all_samples__laplacian_b = []
        all_samples__db_dt = []
        

        # Go over each sample
        for j, sample in enumerate(grid_offsets):
            offset = torch.floor(sample*self.params.resolution_factor)/self.params.resolution_factor

            # Interpolate spline coefficients to obtain the necessary quantities
            h, grad_h, dh_dt, u, grad_u, laplacian_u, du_dt, v, grad_v, laplacian_v, dv_dt, s, grad_s, laplacian_s, ds_dt, b, grad_b, laplacian_b, db_dt = self.dataset.interpolate_states(old_hidden_state, new_hidden_state, offset)

            # Append these samples to the respective lists
            all_samples__h.append(h)
            all_samples__grad_h.append(grad_h)
            all_samples__dh_dt.append(dh_dt)
            all_samples__u.append(u)
            all_samples__grad_u.append(grad_u)
            all_samples__laplacian_u.append(laplacian_u)
            all_samples__du_dt.append(du_dt)
            all_samples__v.append(v)
            all_samples__grad_v.append(grad_v)
            all_samples__laplacian_v.append(laplacian_v)
            all_samples__dv_dt.append(dv_dt)
            all_samples__s.append(s)
            all_samples__grad_s.append(grad_s)
            all_samples__laplacian_s.append(laplacian_s)
            all_samples__ds_dt.append(ds_dt)
            all_samples__b.append(b)
            all_samples__grad_b.append(grad_b)
            all_samples__laplacian_b.append(laplacian_b)
            all_samples__db_dt.append(db_dt)

        # Stack all interpolations in the right order
        h =             torch.cat(all_samples__h, dim=0)
        grad_h =        torch.cat(all_samples__grad_h, dim=0)
        dh_dt =         torch.cat(all_samples__dh_dt, dim=0)
        u =             torch.cat(all_samples__u, dim=0)
        grad_u =        torch.cat(all_samples__grad_u, dim=0)
        laplacian_u =   torch.cat(all_samples__laplacian_u, dim=0)
        du_dt =         torch.cat(all_samples__du_dt, dim=0)
        v =             torch.cat(all_samples__v, dim=0)
        grad_v =        torch.cat(all_samples__grad_v, dim=0)
        laplacian_v =   torch.cat(all_samples__laplacian_v, dim=0)
        dv_dt =         torch.cat(all_samples__dv_dt, dim=0)
        s =             torch.cat(all_samples__s, dim=0)
        grad_s =        torch.cat(all_samples__grad_s, dim=0)
        laplacian_s =   torch.cat(all_samples__laplacian_s, dim=0)
        ds_dt =         torch.cat(all_samples__ds_dt, dim=0)
        b =             torch.cat(all_samples__b, dim=0)
        grad_b =        torch.cat(all_samples__grad_b, dim=0)
        laplacian_b =   torch.cat(all_samples__laplacian_b, dim=0)
        db_dt =         torch.cat(all_samples__db_dt, dim=0)

        #
        # PHYSICS INFORMED LOSS CALCULATION
        #

        # Add mean water level height
        h_before_relu = h
        h = F.softplus(h - self.params.Hc) + self.params.Hc

        # Create a mask that captures dry plains in the domain
        dry_mask = torch.zeros_like(h)
        dry_mask[torch.where(h_before_relu < self.params.Hc)] = 1

        #
        # Derive bed friction coefficients
        #

        # n: Manning's coefficient
        n = self.params.nb  + (self.params.nv - self.params.nb) * b / self.params.k

        # Cz: Chezy coefficient
        chezy = (1.0 / n) * torch.pow(h, 1.0 / 6.0)

        # Bed friction components
        # Add really small value to u2+v2 to prevent dividing by zero in backprop (deriv of sqroot is 1/sqrt)
        tau_precalc = (self.params.grav / torch.pow(chezy, 2)) * torch.pow(torch.pow(u, 2) + torch.pow(v, 2) + 1e-12, 0.5)
        tau_bx_per_rho = tau_precalc * u
        tau_by_per_rho = tau_precalc * v
        tau_b_per_rho  = (self.params.grav / torch.pow(chezy, 2)) * (torch.pow(u, 2) + torch.pow(v, 2))

        # Effective water height
        he = h - self.params.Hc

        # Compute topographic diffusion term (\/(Ds\/S)) (Ds is a field)
        Ds = self.params.D0 * (1.0 - self.params.pD * (b / self.params.k))
        grad_Ds = - ((self.params.D0 * self.params.pD) / self.params.k) * grad_b

        gradDs_dot_gradS = grad_Ds[:, 1:2] * grad_s[:, 1:2] + grad_Ds[:, 0:1] * grad_s[:, 0:1]

        # Full divergence term
        div_Ds_grad_S = Ds * laplacian_s + gradDs_dot_gradS

        #
        # COMPUTE SAMPLE LOSS
        #

        # h-loss
        loss_h = torch.mean(self.loss_function(
            dh_dt + (u*grad_h[:,1:2] + h_before_relu*grad_u[:,1:2]) + (v*grad_h[:,0:1] + h_before_relu*grad_v[:,0:1])
        ), dim)

        # Momentum loss
        loss_u = torch.mean(self.loss_function(
            du_dt + self.params.grav*(grad_h[:,1:2] + grad_s[:,1:2]) + u*grad_u[:,1:2] + v*grad_u[:,0:1] + tau_bx_per_rho/h - self.params.Du * laplacian_u
        ), dim)

        loss_v = torch.mean(self.loss_function(
            dv_dt + self.params.grav*(grad_h[:,0:1] + grad_s[:,0:1]) + u*grad_v[:,1:2] + v*grad_v[:,0:1] + tau_by_per_rho/h - self.params.Du * laplacian_v
        ), dim)

        # Sediment loss
        if self.training_sediment:
            loss_s = torch.mean(self.loss_function(
                ds_dt - self.params.Sin * (he / (self.params.Qs + he)) + self.params.Es * (1.0 - self.params.pE * (b/self.params.k)) * s * tau_b_per_rho - div_Ds_grad_S
            ), dim)
        else:
            loss_s = torch.zeros_like(loss_h)

        # Vegetation loss
        if self.training_vegetation:
            loss_b = torch.mean(self.loss_function(
                db_dt - self.params.r * b * (1.0 - (b/self.params.k)) * (self.params.Qq / (self.params.Qq + he)) + self.params.EB * b * tau_b_per_rho - self.params.DB * laplacian_b
            ), dim)
        else:
            loss_b = torch.zeros_like(loss_h)

        #
        # Boundary condition loss
        #

        # Closed boundary
        loss_bound_closed = torch.mean(sample_closed_mask[:,:,1:-1,1:-1] * self.loss_function(
            grad_h
        ), dim)

        loss_bound_closed = loss_bound_closed + torch.mean(sample_closed_mask[:,:,1:-1,1:-1] * self.loss_function(
            u
        ), dim)

        loss_bound_closed = loss_bound_closed + torch.mean(sample_closed_mask[:,:,1:-1,1:-1] * self.loss_function(
            v
        ), dim)

        if self.training_sediment:
            loss_bound_closed = loss_bound_closed + torch.mean(sample_closed_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_s
            ), dim)

        if self.training_vegetation:
            loss_bound_closed = loss_bound_closed + torch.mean(sample_closed_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_b / self.params.k  # Normalize any loss related to b by max. carrying capacity to match scale-difference
            ), dim)

        # Open boundary
        loss_bound_open = torch.mean(sample_opened_mask[:,:,1:-1,1:-1] * self.loss_function(
            grad_h
        ), dim)

        # loss_bound_open = loss_bound_open + torch.mean(sample_opened_mask[:,:,1:-1,1:-1] * self.loss_function(
        #     laplacian_u
        # ), dim)

        # loss_bound_open = loss_bound_open + torch.mean(sample_opened_mask[:,:,1:-1,1:-1] * self.loss_function(
        #     laplacian_v
        # ), dim)

        if self.training_sediment:
            loss_bound_open = loss_bound_open + torch.mean(sample_opened_mask[:,:,1:-1,1:-1] * self.loss_function(
                s # S = 0
            ), dim)

        if self.training_vegetation:
            loss_bound_open = loss_bound_open + torch.mean(sample_opened_mask[:,:,1:-1,1:-1] * self.loss_function(
                grad_b / self.params.k  # Normalize any loss related to b by max. carrying capacity to match scale-difference
            ), dim)

        # Water level BC loss
        loss_bound_h = torch.mean(sample_h_mask[:,:,1:-1,1:-1] * self.loss_function(
            h_before_relu - sample_h_cond[:,:,1:-1,1:-1]
        ), dim)

        # Auxilary boundary loss
        loss_bound_aux = torch.mean(dry_mask * self.loss_function(
            u
        ), dim)

        loss_bound_aux = loss_bound_aux + torch.mean(dry_mask * self.loss_function(
            v
        ), dim)

        loss_bound = loss_bound_closed + loss_bound_open + loss_bound_h + loss_bound_aux

        # Normalize towards the number of samples taken
        loss_h = loss_h / self.params.n_samples
        loss_u = loss_u / self.params.n_samples
        loss_v = loss_v / self.params.n_samples
        loss_s = loss_s / self.params.n_samples
        loss_b = loss_b / self.params.n_samples
        loss_bound = loss_bound / self.params.n_samples

        # Normalize vegetation loss for scale-difference
        loss_b = loss_b / self.params.k**2

        return loss_h, loss_u, loss_v, loss_s, loss_b, loss_bound

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
        self.optimizer = Adam([
            {"params": self.water_net.parameters(), "lr": self.params.lr},
            # {"params": self.sediment_net.parameters(), "lr": self.params.lr},
            # {"params": self.vegetation_net.parameters(), "lr": self.params.lr},
        ])
        self.optimizer = PCGrad(self.optimizer)

        # torch.autograd.set_detect_anomaly(True)

        #
        # Logger
        #
        self.logger = Logger(parameters.get_description(self.params), use_csv=self.params.log_csv, use_tensorboard=self.params.log_tensorboard)
        if self.params.load_latest or self.params.load_date_time is not None or self.params.load_index is not None:

            # raise Exception("NotImplementedException: Loading state for training is not yet supported")

            self.load_logger = Logger(parameters.get_description(self.params), use_csv=self.params.log_csv, use_tensorboard=self.params.log_tensorboard)
            if self.params.load_optimizer:
                self.params.load_date_time, self.params.load_index = self.logger.load_state("water_net", self.water_net, self.optimizer.optimizer,
                                                                                  self.params.load_date_time,
                                                                                  self.params.load_index)

                self.params.load_date_time, self.params.load_index = self.logger.load_state("sediment_net", self.sediment_net, self.optimizer.optimizer,
                                                                                  self.params.load_date_time,
                                                                                  self.params.load_index)

                self.params.load_date_time, self.params.load_index = self.logger.load_state("vegetation_net", self.vegetation_net, self.optimizer.optimizer,
                                                                                  self.params.load_date_time,
                                                                                  self.params.load_index)

                print(f"\nCONFIGURATION LOADED OPTIMIZER\n")
            else:
                self.params.load_date_time, self.params.load_index = self.logger.load_state("water_net", self.water_net, None, self.params.load_date_time,
                                                                                  self.params.load_index)

                self.params.load_date_time, self.params.load_index = self.logger.load_state("sediment_net", self.sediment_net, None, self.params.load_date_time,
                                                                                  self.params.load_index)

                self.params.load_date_time, self.params.load_index = self.logger.load_state("vegetation_net", self.vegetation_net, None, self.params.load_date_time,
                                                                                  self.params.load_index)


            self.params.load_index = int(self.params.load_index)
            print(f"PRELOAD OF THE PREVIOUS LEARNED STATE IS ENABLED!!! \n\n\nloaded: {self.params.load_date_time}, {self.params.load_index} \n\n\nPRELOAD OF THE PREVIOUS LEARNED STATE IS ENABLED!!!")

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
        self.water_net.train()
        self.sediment_net.train()
        self.vegetation_net.train()


        # Record starting time
        self.train_start_time = time.time()


        # Training loop:
        # Start from the most recently finished epoch and train until the configured number
        # of epochs has been reached.
        for epoch in range(self.params.load_index, self.params.n_epochs):

            # Each epoch consists of a configurable number of batches.
            for i in range(self.params.n_batches_per_epoch):

                # After 2 epochs of training, start prioritizing training the hydrodynamics over sediment and vegetation
                # S & B converge much quicker, therefore after 2 epochs we can start only training S & B every so many iterations
                # if epoch >= 2:
                #     self.training_sediment = False
                #     self.training_vegetation = False

                #     if i % 5 == 0:
                #         self.training_sediment = True
                #         self.training_vegetation = True

                # Ask for a batch from the dataset
                old_hidden_state, closed_mask, opened_mask, h_mask, h_cond, grid_offsets, sample_closed_masks, sample_opened_masks, sample_h_masks, sample_h_conds = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                # Water
                new_hidden_state_water = self.water_net(old_hidden_state, closed_mask, opened_mask, h_mask, h_cond)

                # Sediment
                if self.training_sediment:
                    new_hidden_state_sediment = self.sediment_net(old_hidden_state, closed_mask, opened_mask)
                else:
                    new_hidden_state_sediment = self.dataset.variables.extract_from(old_hidden_state, "s")

                # Vegetation
                if self.training_vegetation:
                    new_hidden_state_vegetation = self.vegetation_net(old_hidden_state, closed_mask, opened_mask)
                else:
                    new_hidden_state_vegetation = self.dataset.variables.extract_from(old_hidden_state, "b")

                # Compile the full new hidden state
                new_hidden_state = torch.cat([new_hidden_state_water, new_hidden_state_sediment, new_hidden_state_vegetation], dim=1)

                dim = [1,2,3]
                loss_h, loss_u, loss_v, loss_s, loss_b, loss_bound = self.compute_batch_loss(old_hidden_state, new_hidden_state, grid_offsets, sample_closed_masks, sample_opened_masks, sample_h_masks, sample_h_conds, dim)

                # Compute the mean loss
                loss_h = torch.mean(loss_h)
                loss_u = torch.mean(loss_u)
                loss_v = torch.mean(loss_v)
                loss_s = torch.mean(loss_s)
                loss_b = torch.mean(loss_b)
                loss_bound = torch.mean(loss_bound)

                # Multiply with the loss weights
                loss_h = loss_h * self.params.loss_h
                loss_u = loss_u * self.params.loss_momentum
                loss_v = loss_v * self.params.loss_momentum
                loss_s = loss_s * self.params.loss_s
                loss_b = loss_b * self.params.loss_b
                loss_bound = loss_bound * self.params.loss_bound

                # Log loss (per term)
                if self.params.log_loss:
                    loss_h = torch.log(loss_h + 0.0001) # Add small epsilon to prevent -inf loss due to log
                    loss_u = torch.log(loss_u + 0.0001)
                    loss_v = torch.log(loss_v + 0.0001)
                    loss_s = torch.log(loss_s + 0.0001)
                    loss_b = torch.log(loss_b + 0.0001)
                    loss_bound = torch.log(loss_bound + 0.0001)


                # For backprop using PCGrad, construct each loss term
                pcgrad_losses = [
                    loss_h,
                    loss_u,
                    loss_v,
                    loss_bound
                ]

                # In the sediment training stage, add sediment to PCGrad as well
                if self.training_sediment:
                    pcgrad_losses.append(loss_s)

                if self.training_vegetation:
                    pcgrad_losses.append(loss_b)

                # Reset old gradients to 0 and compute new gradients with backpropagation
                self.water_net.zero_grad()
                self.sediment_net.zero_grad()
                self.vegetation_net.zero_grad()
                
                # PCGrad backprop pass
                self.optimizer.pc_backward(pcgrad_losses)

                # Clip gradients
                # if self.params.clip_grad_value is not None:
                #     torch.nn.utils.clip_grad_value_(self.net.parameters(),self.params.clip_grad_value)

                # if self.params.clip_grad_norm is not None:
                #     torch.nn.utils.clip_grad_norm_(self.net.parameters(),self.params.clip_grad_norm)
                
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
                    self.logger.log("loss_u", loss_u.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_v", loss_v.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_s", loss_s.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_b", loss_b.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)
                    self.logger.log("loss_bound", loss_bound.detach().cpu(), epoch * self.params.n_batches_per_epoch + i)


            # Print the average time per iteration
            print(f"Average time per batch: {((time.time() - self.train_start_time)/((epoch+1)*self.params.n_batches_per_epoch)):.2f} seconds")

            # Save the training state after each epoch
            if self.params.log:
                self.logger.save_state("water_net", self.water_net, self.optimizer, epoch + 1)
                # self.logger.save_state("sediment_net", self.sediment_net, self.optimizer, epoch + 1)
                # self.logger.save_state("vegetation_net", self.vegetation_net, self.optimizer, epoch + 1)


    def visualize(self, window, print_loss_images=False):
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
        date_time, index = self.logger.load_state("water_net", self.water_net, None, datetime=self.params.load_date_time, index=self.params.load_index)

        try:
            date_time, index = self.logger.load_state("sediment_net", self.sediment_net, None, datetime=self.params.load_date_time, index=self.params.load_index)
            self.training_sediment = True
        except:
            self.training_sediment = False
        
        try:
            date_time, index = self.logger.load_state("vegetation_net", self.vegetation_net, None, datetime=self.params.load_date_time, index=self.params.load_index)
            self.training_vegetation = True
        except:
            self.training_vegetation = False

        # Load loss progression in pandas dataframe
        training_loss = self.logger.load_logs("loss_h", "loss_u", "loss_v", "loss_bound", datetime=self.params.load_date_time)

        # Enable evaluation of the model
        self.water_net.eval()
        self.sediment_net.eval()
        self.vegetation_net.eval()

        print(f"Loaded {self.params.net}: {date_time}, index: {index}")

        # Open the visualization window
        window.set_training_loss(training_loss)
        window.open()

        

        # Simulation loop
        def simulation_loop():
            sim_index = 0
            
            while window.is_open:

                # Ask for a batch from the dataset
                old_hidden_state, closed_mask, opened_mask, h_mask, h_cond, grid_offsets, sample_closed_masks, sample_opened_masks, sample_h_masks, sample_h_conds = self.dataset.ask()

                # Predict the new domain state by performing a forward pass through the network
                # Water
                new_hidden_state_water = self.water_net(old_hidden_state, closed_mask, opened_mask, h_mask, h_cond)

                # Sediment
                if self.training_sediment:
                    new_hidden_state_sediment = self.sediment_net(old_hidden_state, closed_mask, opened_mask)
                else:
                    new_hidden_state_sediment = self.dataset.variables.extract_from(old_hidden_state, "s")

                # Vegetation
                if self.training_vegetation:
                    new_hidden_state_vegetation = self.vegetation_net(old_hidden_state, closed_mask, opened_mask)
                else:
                    new_hidden_state_vegetation = self.dataset.variables.extract_from(old_hidden_state, "b")

                # Compile the full new hidden state
                new_hidden_state = torch.cat([new_hidden_state_water, new_hidden_state_sediment, new_hidden_state_vegetation], dim=1)

                # Store the newly obtained result in the dataset
                self.dataset.tell(new_hidden_state)

                if sim_index % window.interval == 0:

                    if print_loss_images:
                        dim = [0, 1]
                        loss_h, loss_u, loss_v, loss_s, loss_b, loss_bound = self.compute_batch_loss(old_hidden_state, new_hidden_state, grid_offsets, sample_closed_masks, sample_opened_masks, sample_h_masks, sample_h_conds, dim)
                        
                        def __norm(loss_image):
                            loss_image = loss_image - torch.min(loss_image)
                            loss_image = loss_image / torch.max(loss_image)
                            loss_image = torch.log(loss_image)
                            return loss_image
                    
                        # Scale the loss so they can be projected in the images
                        loss_h = __norm(loss_h)
                        loss_u = __norm(loss_u)
                        loss_v = __norm(loss_v)
                        loss_s = __norm(loss_s)
                        loss_b = __norm(loss_b)
                        
                        window.set_data(loss_h, loss_u, loss_v, loss_s, loss_b, sim_index)

                    else:
                        # Interpolate spline coefficients to obtain the necessary quantities
                        h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = self.dataset.interpolate_superres(old_hidden_state, self.params.resolution_factor)
    
                        # Display water level thickness h
                        window.set_data(h[0,0], u[0,0], v[0,0], s[0,0], b[0,0], sim_index)
                        # window.append_loss(loss_h, loss_u, loss_v, loss_s, loss_b)
    
                        # Uncomment the following line to store images in their respective ablation study folder:
                        # torch.save(torch.cat([h, u, v, s, b], dim=1), f"./ablation_output/ablation {self.params.ablation_model}/iteration_{sim_index}.pt")
                

                sim_index += 1

                print(f"\rIteration {sim_index}", end="")

        # Start the simulation thread
        sim_thread = threading.Thread(target=simulation_loop)
        sim_thread.start()

        # Window main thread until closed
        while window.is_open:
            window.update()

        # Join the sim thread
        sim_thread.join()


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

