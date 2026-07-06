import torch
import torch.nn.functional as F
from window import MultiWindow
from ssim import *
import matplotlib.pyplot as plt


class SaltmarshDomain:

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
        self.s = torch.zeros(1, 1, self.width, self.height).to(self.device)
        self.b = torch.zeros(1, 1, self.width, self.height).to(self.device)

        # Default parameters
        self.morphacc = 44_712
        self.D0 = 1e-7
        self.DB = 6e-9
        self.DU = 0.5
        self.ES = 2.5e-4
        self.EB = 1e-5
        self.grav = 9.81
        self.H0 = 0.02
        self.Hc = 1e-3
        self.Hin = 1e-5
        self.k = 1500
        self.nb = 0.016
        self.nv = 0.2
        self.pD = 0.99
        self.pE = 0.9
        self.pest = 0.002
        self.Qq = 0.02
        self.Qs = 6e-4
        self.r = 3.2e-8
        self.Sin = 5e-9

        # Initial conditions
        self.h[:,:,:,:] = self.H0

        vegetation_random = torch.rand_like(self.b)
        self.b[torch.where(vegetation_random <= self.pest)] = self.k

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

    
    def simulate(self, dt=0.0125):
        
        # Copy the current state
        h = self.h
        u = self.u
        v = self.v
        s = self.s
        b = self.b

        # Estimate the amount of outflow as the volume of water flowing per area per second of the outflow boundary
        outflow_per_cell = h[:,:,:,-1] * u[:,:,:,-1] * self.dy # Water drainage in m3/s per cell
        water_drained = torch.sum(outflow_per_cell * dt) # Total water drained in m3 in this timestep

        # Estimate the volume of water that's added to the domain by capping at Hc
        water_compensation = torch.sum(h[torch.where(h < self.Hc)]) * self.dx * self.dy

        # Compute Hin addition of water in m3
        water_Hin = self.Hin * self.width * self.height * self.dx * self.dy

        # 1. Ensure positivity of water layer thickness
        h = torch.clamp(h, min=self.Hc)

        # 2. Update flow velocities
        # 2.1 Compute Manning's coefficient, Chezy coefficient and bed shear stress components tau_bx, tau_by and tau_b
        n = self.nb + (self.nv - self.nb) * (b / self.k)
        Cz = (1.0 / n) * torch.pow(h, 1.0/6.0)
        tau_precalc = (self.grav / (torch.pow(Cz, 2))) * torch.pow(torch.pow(u, 2) + torch.pow(v, 2), 0.5)
        tau_bx_per_rho = tau_precalc * u
        tau_by_per_rho = tau_precalc * v
        tau_b_per_rho = (self.grav / (torch.pow(Cz, 2))) * (torch.pow(u, 2) + torch.pow(v, 2))

        # 2.2 Velocity update step
        du_dt = -self.grav * self.d_dx(h + s) - u * self.d_dx(u) - v * self.d_dy(u) - tau_bx_per_rho/h + self.DU * (self.d2_dx2(u) + self.d2_dy2(u))
        dv_dt = -self.grav * self.d_dy(h + s) - u * self.d_dx(v) - v * self.d_dy(v) - tau_by_per_rho/h + self.DU * (self.d2_dx2(v) + self.d2_dy2(v))

        u = u + du_dt * dt
        v = v + dv_dt * dt

        # 2.3 Enforce BCs for u and v
        # Closed boundary left
        u[:,:,:,0] = -u[:,:,:,1]
        v[:,:,:,0] =  v[:,:,:,1]

        # Closed boundary up
        u[:,:,0,:] =  u[:,:,1,:]
        v[:,:,0,:] = -v[:,:,1,:]

        # Closed boundary down
        u[:,:,-1,:] =  u[:,:,-2,:]
        v[:,:,-1,:] = -v[:,:,-2,:]

        # Open boundary right
        u[:,:,:,-1] = 2*u[:,:,:,-2] - u[:,:,:,-3]
        v[:,:,:,-1] = v[:,:,:,-2]

        # 3. Compute update for h
        dh_dt = - self.d_dx(u*h) - self.d_dy(v*h) + self.Hin

        # 4. Compute update for s
        # 4.1 Effective water layer thickness
        he = h - self.Hc

        # 4.2 Sediment diffusivity Ds
        DS = self.D0 * (1 - self.pD *(b / self.k))

        ds_dt = self.Sin * (he / (self.Qs + he)) - self.ES * (1 - self.pE * (b / self.k)) * s * tau_b_per_rho + (self.d_dx(DS * self.d_dx(s)) + self.d_dy(DS * self.d_dy(s)))

        # Compute update for b
        db_dt = self.r * b * (1 - (b / self.k)) * (self.Qq / (self.Qq + he)) - self.EB * b * tau_b_per_rho + self.DB * (self.d2_dx2(b) + self.d2_dy2(b))

        # Update h, s, and b
        h = h + dh_dt * dt
        s = s + ds_dt * dt * self.morphacc
        b = b + db_dt * dt * self.morphacc

        # 5. Enforce BCs on h, s and b

        # h zerograd everywhere
        h[:,:,:,0] = h[:,:,:,1]
        h[:,:,:,-1] = h[:,:,:,-2]
        h[:,:,0,:] = h[:,:,1,:]
        h[:,:,-1,:] = h[:,:,-2,:]

        # s zerograd everywhere except open bound
        s[:,:,:,0] = s[:,:,:,1]
        s[:,:,:,-1] = 0
        s[:,:,0,:] = s[:,:,1,:]
        s[:,:,-1,:] = s[:,:,-2,:]

        # b zerograd everywhere
        b[:,:,:,0] = b[:,:,:,1]
        b[:,:,:,-1] = b[:,:,:,-2]
        b[:,:,0,:] = b[:,:,1,:]
        b[:,:,-1,:] = b[:,:,-2,:]

        # Update domain state
        self.h = h
        self.u = u
        self.v = v
        self.s = s
        self.b = b

        # Update time
        self.t += dt


        return water_Hin, water_compensation.cpu(), water_drained.cpu()





if __name__ == "__main__":
    
    window = MultiWindow(400, 400)

    basin = SaltmarshDomain(400, 400, torch.device("mps"))

    water_source_Hin = []
    water_source_comp = []
    water_drainage = []

    water_source_Hin_plot, = window.axs[1, 2].plot([], [], label="Hin")
    water_source_comp_plot, = window.axs[1, 2].plot([], [], label="comp")
    water_drainage_plot, = window.axs[1, 2].plot([], [], label="drainage")

    # Open the window
    window.open()

    iteration_count = 0

    # As long as the window is open, run the simulation
    while window.is_open:

        # Make a simulation step
        source_Hin, source_comp, drainage = basin.simulate()

        # Collect the water in and outflows
        water_source_Hin.append(source_Hin)
        water_source_comp.append(source_comp)
        water_drainage.append(drainage)

        if iteration_count % 1000 == 0:
            # Update the window state
            window.set_data(
                basin.h[0, 0],
                basin.u[0, 0],
                basin.v[0, 0],
                basin.s[0, 0],
                basin.b[0, 0]
            )

            water_source_Hin_plot.set_xdata(range(len(water_source_Hin)))
            water_source_Hin_plot.set_ydata(water_source_Hin)
            water_source_comp_plot.set_xdata(range(len(water_source_comp)))
            water_source_comp_plot.set_ydata(water_source_comp)
            water_drainage_plot.set_xdata(range(len(water_drainage)))
            water_drainage_plot.set_ydata(water_drainage)

            window.axs[1, 2].relim()
            window.axs[1, 2].autoscale_view()
            window.axs[1, 2].legend()

            window.update()

        iteration_count += 1
