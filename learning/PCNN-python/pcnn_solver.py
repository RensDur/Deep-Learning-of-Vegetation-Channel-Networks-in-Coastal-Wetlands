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

    def step_loss(self):
        h, u, v, S, B = self.dataset.ask()





