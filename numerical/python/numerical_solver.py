import torch
import torch.nn as nn
from domain import Domain

class NumericalSolver:


    def __init__(self):
        self.device = torch.device('cpu') # Default to CPU
        # Switch to MPS (Apple Metal) if available
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Or CUDA if we're on an Nvidia machine
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')

        print(f"Using torch device '{self.device}'")

        # Domain
        self.domain = Domain(200, 200, 0.05)
        self.domain.init_preset_gaussian_wave(5, 5, 1, 1)
        self.domain.to(self.device)

        # Timestep
        self.dt = 0.0005

        # Parepare torch models
        self.dx_kernel = torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], dtype=torch.float32, device=self.device)
        self.dx_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', bias=False).to(self.device)
        self.dx_conv.weight = torch.nn.Parameter(self.dx_kernel)

        self.dy_kernel = torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]], dtype=torch.float32, device=self.device)
        self.dy_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', bias=False).to(self.device)
        self.dy_conv.weight = torch.nn.Parameter(self.dy_kernel)

        self.dx2_kernel = torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]], dtype=torch.float32, device=self.device)
        self.dx2_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', bias=False).to(self.device)
        self.dx2_conv.weight = torch.nn.Parameter(self.dx2_kernel)

        self.dy2_kernel = torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]], dtype=torch.float32, device=self.device)
        self.dy2_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', bias=False).to(self.device)
        self.dy2_conv.weight = torch.nn.Parameter(self.dy2_kernel)


    def d_dx(self, quantity):
        # Pack and unpack the 'quantity' tensor to match expected size by the convolution,
        # then divide by twice the dx of our domain, because we take the derivative over this cell, using both neighbors.
        return self.dx_conv(quantity.unsqueeze(0)).detach().squeeze() / (2*self.domain.dx)

    def d_dy(self, quantity):
        # Pack and unpack the 'quantity' tensor to match expected size by the convolution,
        # then divide by twice the dy of our domain, because we take the derivative over this cell, using both neighbors.
        return self.dy_conv(quantity.unsqueeze(0)).detach().squeeze() / (2*self.domain.dy)

    def d2_dx2(self, quantity):
        # Second partial derivative central difference
        return self.dx2_conv(quantity.unsqueeze(0)).detach().squeeze() / (self.domain.dx**2)

    def d2_dy2(self, quantity):
        # Second partial derivative central difference
        return self.dy2_conv(quantity.unsqueeze(0)).detach().squeeze() / (self.domain.dy**2)

    def solve_step(self):
        
        # Compute uh and vh
        uh = self.domain.u * self.domain.h
        vh = self.domain.v * self.domain.h

        # Convolution with 3x3 kernel and padding
        # to compute d(uh)/dx and d(vh)/dy
        duh_dx = self.d_dx(uh)

        # Divide dvhy by dy to obtain d(vh)/dy
        dvh_dy = self.d_dy(vh)

        # Compute dh/dt
        dh_dt = -duh_dx - dvh_dy #+ self.domain.Hin

        # Multiply by timestep to obtain dh and compute new h
        dh = dh_dt * self.dt

        # Wetting-drying
        h_updated = torch.clamp(self.domain.h + dh, min=self.domain.Hc)

        #
        # Momentum equations
        #

        # Compute d(h+S)/dx
        eta = self.domain.h + self.domain.S

        deta_dx = self.d_dx(eta)
        deta_dy = self.d_dy(eta)

        du_dx = self.d_dx(self.domain.u)
        du_dy = self.d_dy(self.domain.u)

        dv_dx = self.d_dx(self.domain.v)
        dv_dy = self.d_dy(self.domain.v)

        du_dt = -self.domain.grav * deta_dx - self.domain.u * du_dx - self.domain.v * du_dy
        dv_dt = -self.domain.grav * deta_dy - self.domain.u * dv_dx - self.domain.v * dv_dy

        du = du_dt * self.dt
        dv = dv_dt * self.dt

        u_updated = self.domain.u + du
        v_updated = self.domain.v + dv

        #
        # Enforce boundary conditions
        #

        # dv must be zero or positive on the top boundary
        v_updated[0, :] = torch.clamp(v_updated[0, :], min=0)
        v_updated[-1, :] = torch.clamp(v_updated[-1, :], max=0)
        u_updated[:, 0] = torch.clamp(u_updated[:, 0], min=0)
        u_updated[:, -1] = torch.clamp(u_updated[:, -1], max=0)

        # Update the domain
        self.domain.h = h_updated
        self.domain.u = u_updated
        self.domain.v = v_updated






                
