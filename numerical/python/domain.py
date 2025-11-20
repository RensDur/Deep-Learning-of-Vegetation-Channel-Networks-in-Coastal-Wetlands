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
        self.h = torch.zeros(self.height, self.width)
        self.S = torch.zeros(self.height, self.width)
        self.u = torch.zeros(self.height, self.width)
        self.v = torch.zeros(self.height, self.width)
        self.B = torch.zeros(self.height, self.width)

        self.Hin = 1e-5
        self.Hc = 1e-3
        self.H0 = 0.02 # Initial water thickness
        self.grav = 9.81
        self.rho = 1000 # Water density
        self.Du = 0.5 # Turbulent Eddy velocity
        self.nb = 0.016 # bed roughness for bare land
        self.nv = 0.2 # bed roughness for vegetated land
        self.k = 1500 # Vegetation carrying capacity
        self.D0 = 1e-7 # Sediment diffusivity in absence of vegetation
        self.pD = 0.99 # fraction by which sediment diffusivity is reduced when vegetation is at carrying capacity
        self.Sin = 5e-9 # Maximum sediment input rate
        self.Qs = 6e-4 # water layer thickness at which sediment input is halved
        self.Es = 2.5e-4 # Sediment erosion rate
        self.pE = 0.9 # Fraction by which sediment erosion is reduced when vegetation is at carrying capacity
        self.r = 3.2e-8 # Intrinsic plant growth rate (=1 per year)
        self.Qq = 0.02 # Water layer thickness at which vegetation growth is halved
        self.EB = 1e-5 # Vegetation erosion rate
        self.DB = 6e-9 # Vegetation diffusivity
        self.morphological_acc_factor = 44712 # Morphological acceleration factor, required for S and B
        self.pEst = 0.002 # Probability of vegetation seedling establishment


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

        self.h[:, :] = 1 + A * torch.exp(
            - (1/(2*(1-correlation**2))) * (((x-x_mu)/x_sig)**2 - 2*correlation*((x-x_mu)/x_sig)*((y-y_mu)/y_sig) + ((y-y_mu)/y_sig)**2)
        )

    def initialize(self):
        # Sedimentary elevation is zero everywhere
        self.S[:, :] = 0

        # Flow velocities are zero everywhere
        self.u[:, :] = 0
        self.v[:, :] = 0

        # Uniform water thickness H0
        self.h[:, :] = self.H0

        # Vegetation density is zero everywhere, except for some randomly placed tussocks
        self.B[:, :] = 0
        self.B[torch.where(torch.rand(self.height, self.width) < self.pEst)] = self.k

    #
    # MEMORY MANAGEMENT
    #
    def to(self, device):
        self.h = self.h.to(device)
        self.S = self.S.to(device)
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.B = self.B.to(device)

    def get_h(self):
        return self.h.cpu().numpy()

    def get_S(self):
        return self.S.cpu().numpy()

    def get_B(self):
        return self.B.cpu().numpy()

    def get_u(self):
        return self.u.cpu().numpy()

    def get_v(self):
        return self.v.cpu().numpy()
