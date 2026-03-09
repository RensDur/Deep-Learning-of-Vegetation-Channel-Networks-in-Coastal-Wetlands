import torch
from dataset import Dataset

class Solver:

    def __init__(self, width=200, height=200, device=torch.device("cpu")):

        self.device = device

        self.width = width
        self.height = height

        self.dx = 1
        self.dy = 1
        self.dt = 0.0001

        self.dataset = Dataset(self.width, self.height, self.device)

        #
        # Convolution kernels
        #

        # First order derivatives
        self.dx_kernel = torch.tensor([-0.5,0,0.5], device=self.device).view(1, 1, 1, 3)
        self.dy_kernel = torch.tensor([-0.5,0,0.5], device=self.device).view(1, 1, 3, 1)

        # Second order derivatives
        self.dx2_kernel = torch.tensor([1.0, -2.0, 1.0], device=self.device).view(1, 1, 1, 3)
        self.dy2_kernel = torch.tensor([1.0, -2.0, 1.0], device=self.device).view(1, 1, 3, 1)

    def d_dx(self, quantity):
        return F.conv2d(quantity, self.dx_kernel, padding=(0,1)) / self.dx

    def d_dy(self, quantity):
        return F.conv2d(quantity, self.dy_kernel, padding=(1,0)) / self.dy

    def d2_dx2(self, quantity):
        return F.conv2d(quantity, self.dx2_kernel, padding=(0,1)) / (self.dx**2)

    def d2_dy2(self, quantity):
        return F.conv2d(quantity, self.dy2_kernel, padding=(1,0)) / (self.dy**2)

    
    def get(self):
        """
        Obtain quantities stored in the dataset
        """
        h, u, v, s = self.dataset.get()

        return h.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), s.cpu().numpy()


    def step(self):
        """
        Numerical solve step (stepsize dt)
        """

        pass