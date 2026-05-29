import torch
import torch.nn.functional as F
from dataset import Dataset

class Solver:

    def __init__(self, width=200, height=200, device=torch.device("cpu")):

        self.device = device

        self.width = width
        self.height = height

        self.dx = 2
        self.dy = 2
        self.dt = 0.001

        self.dataset = Dataset(self.width, self.height, self.device)
        self.dataset.start_condition("rest-lake")

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
        h, hu, hv, s = self.dataset.get()

        return h.detach().cpu().numpy(), hu.detach().cpu().numpy(), hv.detach().cpu().numpy(), s.detach().cpu().numpy()


    def step(self):
        """
        Numerical solve step (stepsize dt)
        """

        g = 9.81


        # Ask for the current environment state
        h, hu, hv, s = self.dataset.get()

        u = hu / h
        v = hv / h

        # 1. Compute updated flow velocities
        dhu_dt = -self.d_dx(h * u**2 + 0.5 * g * h**2) - self.d_dy(h*u*v)
        dhv_dt = -self.d_dy(h * v**2 + 0.5 * g * h**2) - self.d_dx(h*u*v)

        hu += dhu_dt * self.dt
        hv += dhv_dt * self.dt

        u = hu / h
        v = hv / h

        # 2. Apply boundary conditions on flow velocities

        # Left boundary
        hu[:, :, :, 0] = 0.1
        hv[:, :, :, 0] =  hv[:, :, :, 1]

        # Right boundary
        hu[:, :, :, -1] =  hu[:, :, :, -2]
        hv[:, :, :, -1] =  hv[:, :, :, -2]

        # Top boundary
        hu[:, :, 0, :] =  hu[:, :, 1, :]
        hv[:, :, 0, :] = -hv[:, :, 1, :]

        # Bottom boundary
        hu[:, :, -1, :] =  hu[:, :, -2, :]
        hv[:, :, -1, :] = -hv[:, :, -2, :]

        # Obstacle
        # hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2] = -hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2-1]
        # hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2+1] = -hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2+2]

        # hv[:, :, self.height//2-10, (self.width//2-1):(self.width//2+1)] = -hv[:, :, self.height//2-11, (self.width//2-1):(self.width//2+1)]
        # hv[:, :, self.height//2+10, (self.width//2-1):(self.width//2+1)] = -hv[:, :, self.height//2+11, (self.width//2-1):(self.width//2+1)]

        hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2] = 0
        hu[:, :, (self.height//2-10):(self.height//2+10), self.width//2+1] = 0

        hv[:, :, self.height//2-10, (self.width//2-1):(self.width//2+1)] = 0
        hv[:, :, self.height//2+10, (self.width//2-1):(self.width//2+1)] = 0


        # 3. Compute updated water layer thickness
        dh_dt = -self.d_dx(hu) - self.d_dy(hv)

        h += dh_dt * self.dt

        # 4. Apply boundary condition to h
        h[:, :, :, 0] = h[:, :, :, 1]
        h[:, :, :, -1] = h[:, :, :, -2]
        h[:, :, 0, :] = h[:, :, 1, :]
        h[:, :, -1, :] = h[:, :, -2, :]

        h[:, :, (self.height//2-10):(self.height//2+10), self.width//2] = h[:, :, (self.height//2-10):(self.height//2+10), self.width//2-1]
        h[:, :, (self.height//2-10):(self.height//2+10), self.width//2+1] = h[:, :, (self.height//2-10):(self.height//2+10), self.width//2+2]

        h[:, :, self.height//2-10, (self.width//2-1):(self.width//2+1)] = h[:, :, self.height//2-11, (self.width//2-1):(self.width//2+1)]
        h[:, :, self.height//2+10, (self.width//2-1):(self.width//2+1)] = h[:, :, self.height//2+11, (self.width//2-1):(self.width//2+1)]

        # Store in the dataset
        self.dataset.put(h, hu, hv, s)
