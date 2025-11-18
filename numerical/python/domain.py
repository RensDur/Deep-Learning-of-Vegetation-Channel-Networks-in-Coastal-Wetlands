import torch
import math


class Domain:


    def __init__(self, width, height, meter_per_cell):
        self.width = width
        self.height = height

        # Create a torch grid for h, S, u and v
        # h: Water layer thickness
        # S: Sediment bed
        # u: shoreward depth-averaged flow velocity
        # v: alongshore depth-averaged flow velocity
        self.h = torch.zeros(1, 1, self.height, self.width)
        self.S = torch.zeros(1, 1, self.height, self.width)
        self.u = torch.zeros(1, 1, self.height, self.width)
        self.v = torch.zeros(1, 1, self.height, self.width)

        self.Hin = 1e-3
        self.Hc = 1e-5
        self.grav = 9.81

        # Domain properties
        self.dx = meter_per_cell
        self.dy = meter_per_cell
        self.xlen = self.width * self.dx
        self.ylen = self.height * self.dy

    #
    # INITIAL CONDITIONS (PRESETS)
    #
    def init_preset_gaussian_wave(self, x_mu, y_mu, x_sig, y_sig, correlation=0):
        x = torch.linspace(0, self.xlen, self.width)
        y = torch.linspace(0, self.ylen, self.height)
        x, y = torch.meshgrid(x, y, indexing='xy')

        A = 1 / (2*math.pi * x_sig * y_sig * (1-correlation**2)**0.5)

        self.h[0, 0, :, :] = A * torch.exp(
            - (1/(2*(1-correlation**2))) * (((x-x_mu)/x_sig)**2 - 2*correlation*((x-x_mu)/x_sig)*((y-y_mu)/y_sig) + ((y-y_mu)/y_sig)**2)
        )

    #
    # MEMORY MANAGEMENT
    #
    def to(self, device):
        self.h.to(device)
        self.S.to(device)
        self.u.to(device)
        self.v.to(device)

    def get_h(self):
        return self.h[0, 0, :, :].detach().cpu().numpy()
