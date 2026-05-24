import torch
import torch.nn.functional as F
from window import MultiWindow
from ssim import *
import matplotlib.pyplot as plt


class ClosedWaterBasin:

    def __init__(self, width, height, torch_device=torch.device("cpu")):

        # Basin dimensions
        self.width = width
        self.height = height

        self.dx = 1
        self.dy = 1

        # Torch device
        self.device = torch_device

        # Allocate memory for h, u, and v fields and move them to the GPU
        self.h = torch.zeros(1, 1, self.width, self.height).to(self.device)
        self.u = torch.zeros(1, 1, self.width, self.height).to(self.device)
        self.v = torch.zeros(1, 1, self.width, self.height).to(self.device)

        # Default parameters
        self.H0 = 1
        self.g = 9.81
        self.epsilon = 0.001
        self.k = 0.01
        self.nu = 0.5

        # Initial conditions
        self.h[:,:,:,:] = self.H0

        # Keep track of time
        self.t = 0

        #
        # Define convolution operators
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

    
    def simulate(self, dt=0.1):
        
        # Copy the current state
        h = self.h
        u = self.u
        v = self.v

        # Calculate and update the flow velocity components
        du_dt = - self.g * self.d_dx(h) - self.k*u - u * self.d_dx(u) - v * self.d_dy(u) + self.nu * (self.d2_dx2(u) + self.d2_dy2(u))
        dv_dt = - self.g * self.d_dy(h) - self.k*v - u * self.d_dx(v) - v * self.d_dy(v) + self.nu * (self.d2_dx2(v) + self.d2_dy2(v))

        u += du_dt * dt
        v += dv_dt * dt

        # Apply boundary conditions on u and v
        u[:,:,:,0] = -u[:,:,:,1]
        u[:,:,:,-1] = -u[:,:,:,-2]

        v[:,:,0,:] = -v[:,:,1,:]
        v[:,:,-1,:] = -v[:,:,-2,:]

        # Calculate and update the continuity equation
        dh_dt = -self.d_dx(h*u) - self.d_dy(h*v) - self.epsilon*h

        h += dh_dt * dt

        # Apply boundary conditions on h
        h[:,:,:,0] = h[:,:,:,1]
        h[:,:,:,-1] = h[:,:,:,-2]
        h[:,:,0,:] = h[:,:,1,:]
        h[:,:,-1,:] = h[:,:,-2,:]

        oscillator_radius = 10

        for x in range(-oscillator_radius, oscillator_radius+1):
            for y in range(-oscillator_radius, oscillator_radius+1):
                if (x**2 + y**2 <= oscillator_radius**2):
                    h[:,:,self.height//2+y,self.width//2+x] = self.H0 + 0.5*torch.sin(torch.Tensor([self.t]))

        # h[:,:,(self.height//2 - 5):(self.height//2 + 5),(self.width//2-5):(self.width//2+5)] = self.H0 + 0.5 * torch.sin(torch.Tensor([self.t])).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 10, 10)

        # Update the internal state
        self.h = h
        self.u = u
        self.v = v

        self.t += dt





if __name__ == "__main__":
    
    window = MultiWindow(100, 100)

    basin = ClosedWaterBasin(100, 100, torch.device("mps"))
    basin2 = ClosedWaterBasin(100, 100, torch.device("mps"))
    basin2.epsilon = 0.005

    ssim_scores = []
    ssim_plot, = window.axs[1, 2].plot([], [])

    # Open the window
    window.open()

    # As long as the window is open, run the simulation
    while window.is_open:

        # Make a simulation step
        basin.simulate()
        basin2.simulate()

        # Calculate the ssim score
        ssim_spatial = ssim(basin.h, basin2.h)
        ssim_scores.append(torch.mean(ssim_spatial).cpu().item())

        # Update the window state
        window.set_data(
            basin.h[0, 0],
            basin.u[0, 0],
            basin.v[0, 0],
            basin2.h[0, 0],
            ssim_spatial[0, 0]
        )

        ssim_plot.set_xdata(range(len(ssim_scores)))
        ssim_plot.set_ydata(ssim_scores)
        window.axs[1, 2].relim()
        window.axs[1, 2].autoscale_view()

        window.update()
