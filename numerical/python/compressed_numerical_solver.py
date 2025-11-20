import torch
import torch.nn as nn
from compressed_domain import CompressedDomain


class CompressedNumericalSolver:

    def __init__(self):
        self.device = torch.device('cpu')  # Default to CPU
        # Switch to MPS (Apple Metal) if available
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Or CUDA if we're on an Nvidia machine
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')

        print(f"Using torch device '{self.device}'")

        # Domain
        self.domain = CompressedDomain(200, 200, 0.5)
        self.domain.initialize()
        self.domain.to(self.device)

        # Timestep
        self.dt = 0.0125
        self.t = 0

        # Parepare torch models
        self.dx_kernel = torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], dtype=torch.float32, device=self.device)
        self.dx_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', groups=10, bias=False).to(self.device)
        self.dx_conv.weight = torch.nn.Parameter(self.dx_kernel)

        self.dy_kernel = torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]], dtype=torch.float32, device=self.device)
        self.dy_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', groups=10, bias=False).to(self.device)
        self.dy_conv.weight = torch.nn.Parameter(self.dy_kernel)

        self.dx2_kernel = torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]], dtype=torch.float32, device=self.device)
        self.dx2_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', groups=10, bias=False).to(self.device)
        self.dx2_conv.weight = torch.nn.Parameter(self.dx2_kernel)

        self.dy2_kernel = torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]], dtype=torch.float32, device=self.device)
        self.dy2_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', groups=10, bias=False).to(self.device)
        self.dy2_conv.weight = torch.nn.Parameter(self.dy2_kernel)

    def d_dx(self, data):
        # Pack and unpack the 'data' tensor to match expected size by the convolution,
        # then divide by twice the dx of our domain, because we take the derivative over this cell, using both neighbors.
        return self.dx_conv(data.unsqueeze(0)).detach().squeeze() / (2 * self.domain.dx)

    def d_dy(self, data):
        # Pack and unpack the 'data' tensor to match expected size by the convolution,
        # then divide by twice the dy of our domain, because we take the derivative over this cell, using both neighbors.
        return self.dy_conv(data.unsqueeze(0)).detach().squeeze() / (2 * self.domain.dy)

    def d2_dx2(self, data):
        # Second partial derivative central difference
        return self.dx2_conv(data.unsqueeze(0)).detach().squeeze() / (self.domain.dx ** 2)

    def d2_dy2(self, data):
        # Second partial derivative central difference
        return self.dy2_conv(data.unsqueeze(0)).detach().squeeze() / (self.domain.dy ** 2)

    def solve_step(self):

        # Compute d/dx and d/dy for all quantities
        d_dx = self.d_dx(self.domain.data)
        d_dy = self.d_dy(self.domain.data)

        # Compute d2/dx2 and d2/dy2 for all quantities
        d2_dx2 = self.d2_dx2(self.domain.data)
        d2_dy2 = self.d2_dy2(self.domain.data)


