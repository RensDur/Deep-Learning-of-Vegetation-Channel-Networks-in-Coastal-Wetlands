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
        self.domain = Domain(100, 100, 0.05)
        self.domain.init_preset_gaussian_wave(2.5, 2.5, 1, 1)
        self.domain.to(self.device)

        # Timestep
        self.dt = 0.01




    def solve_step(self):

        print("Started to solve step...")
        
        # Compute uh and vh
        uh = self.domain.u * self.domain.h
        vh = self.domain.v * self.domain.h

        # Convolution with 3x3 kernel and padding
        # to compute d(uh)/dx and d(vh)/dy
        dx_kernel = torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], dtype=torch.float32)
        dx_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        dx_conv.weight = torch.nn.Parameter(dx_kernel)

        # Divide duhx by dx to obtain d(uh)/dx
        duh_dx = dx_conv(uh.unsqueeze(0)).detach().squeeze() / self.domain.dx

        dy_kernel = torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]], dtype=torch.float32)
        dy_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        dy_conv.weight = torch.nn.Parameter(dy_kernel)

        # Divide dvhy by dy to obtain d(vh)/dy
        dvh_dy = dy_conv(vh.unsqueeze(0)).detach().squeeze() / self.domain.dy

        # Compute dh/dt
        dh_dt = -duh_dx - dvh_dy + self.domain.Hin

        # Multiply by timestep to obtain dh and compute new h
        dh = dh_dt * self.dt

        # Wetting-drying
        h_updated = torch.clamp(self.domain.h + dh, min=self.domain.Hc)

        print(h_updated)

        #
        # Momentum equations
        #

        # Compute d(h+S)/dx
        eta = self.domain.h + self.domain.S

        deta_dx = dx_conv(eta.unsqueeze(0)).detach().squeeze() / self.domain.dx
        deta_dy = dy_conv(eta.unsqueeze(0)).detach().squeeze() / self.domain.dy

        du_dx = dx_conv(self.domain.u.unsqueeze(0)).detach().squeeze() / self.domain.dx
        du_dy = dy_conv(self.domain.u.unsqueeze(0)).detach().squeeze() / self.domain.dy

        dv_dx = dx_conv(self.domain.v.unsqueeze(0)).detach().squeeze() / self.domain.dx
        dv_dy = dy_conv(self.domain.v.unsqueeze(0)).detach().squeeze() / self.domain.dy

        du_dt = -self.domain.grav * deta_dx - self.domain.u * du_dx - self.domain.v * du_dy
        dv_dt = -self.domain.grav * deta_dy - self.domain.u * dv_dx - self.domain.v * dv_dy

        du = du_dt * self.dt
        dv = dv_dt * self.dt

        u_updated = self.domain.u + du
        v_updated = self.domain.v + dv

        # Update the domain
        self.domain.h[:, :] = h_updated
        self.domain.u[:, :] = u_updated
        self.domain.v[:, :] = v_updated

        print("Finished step!")




                
