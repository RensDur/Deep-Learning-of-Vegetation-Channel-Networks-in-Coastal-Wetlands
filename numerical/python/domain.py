import torch
import math


class Domain:


    def __init__(self, width, height, meter_per_cell):
        self.width = width
        self.height = height

        # Create a torch grid
        self.grid = torch.zeros(self.height, self.width)

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

        self.grid = A * torch.exp(
            - (1/(2*(1-correlation**2))) * (((x-x_mu)/x_sig)**2 - 2*correlation*((x-x_mu)/x_sig)*((y-y_mu)/y_sig) + ((y-y_mu)/y_sig)**2)
        )

    #
    # MEMORY MANAGEMENT
    #
    def to(self, device):
        self.grid.to(device)
    
    #
    # EXPORT TO NUMPY
    #
    def numpy(self):
        return self.grid.numpy()