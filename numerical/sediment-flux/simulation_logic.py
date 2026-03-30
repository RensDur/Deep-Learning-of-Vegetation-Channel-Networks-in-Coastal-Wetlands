import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#
# TORCH DEVICE
#

torch_device = torch.device("cpu")

if torch.backends.mps.is_available():
    torch_device = torch.device("mps")

print(f"Using torch device: {torch_device}")

#
# PARAMETERS
#

H0 = 0.02
Hc = 1e-3
Hin = 1e-5
grav = 9.81

Du = 0.5

D0 = 1e-7
pD = 0.99
pE = 0.9
Sin = 5e-9
Qs = 6e-4
Es = 2.5e-4
nb = 0.016
nv = 0.2

r = 3.2e-8
k = 1500
Qq = 0.02
Eb = 1e-5
Db = 6e-9

dt = 0.0125
dx = 1/4
dy = 1/4

current_t = 0.0

#
# OPERATORS
#
dx_kernel = torch.tensor([-0.5,0,0.5], device=torch_device).view(1, 1, 1, 3)
dy_kernel = torch.tensor([-0.5,0,0.5], device=torch_device).view(1, 1, 3, 1)

# Second order derivatives
dx2_kernel = torch.tensor([1.0, -2.0, 1.0], device=torch_device).view(1, 1, 1, 3)
dy2_kernel = torch.tensor([1.0, -2.0, 1.0], device=torch_device).view(1, 1, 3, 1)

def d_dx(quantity):
    return F.conv2d(quantity, dx_kernel, padding=(0,1)) / dx

def d_dy(quantity):
    return F.conv2d(quantity, dy_kernel, padding=(1,0)) / dy

def d2_dx2(quantity):
    return F.conv2d(quantity, dx2_kernel, padding=(0,1)) / dx**2

def d2_dy2(quantity):
    return F.conv2d(quantity, dy2_kernel, padding=(1,0)) / dy**2


class Solver():


    def __init__(self):
        #
        # ENVIRONMENTS
        #

        self.width = 800
        self.height = 800

        self.h = torch.zeros(1, 1, self.height, self.width)
        self.u = torch.zeros(1, 1, self.height, self.width)
        self.v = torch.zeros(1, 1, self.height, self.width)
        self.s = torch.zeros(1, 1, self.height, self.width)
        self.b = torch.zeros(1, 1, self.height, self.width)

        # H: initial condition
        self.h[:, :, :, :] = H0

        # Randomly place vegetation
        self.b[torch.where(torch.rand_like(self.b) < 0.002)] = k

    #
    # MAIN FUNCTION
    #
    def run_iter(self):

        h = self.h
        u = self.u
        v = self.v
        s = self.s
        b = self.b

        # Manning's n
        n = nb + (nv - nb) * (b / k)

        # Chezys coefficient
        Cz = (1.0 / n) * torch.pow(h, 1.0 / 6.0)

        tau_precalc = (grav / torch.pow(Cz, 2.0)) * torch.pow(torch.pow(u, 2) + torch.pow(v, 2), 0.5)

        tau_bx_per_rho = tau_precalc * u
        tau_by_per_rho = tau_precalc * v
        tau_b_per_rho = (grav / torch.pow(Cz, 2.0)) * (torch.pow(u, 2) + torch.pow(v, 2))

        # Compute flow velocity update
        du_dt = -grav * d_dx(h + s) - u*d_dx(u) - v*d_dy(u) - (tau_bx_per_rho / h) + Du * (d2_dx2(u) + d2_dy2(u))
        dv_dt = -grav * d_dy(h + s) - u*d_dx(v) - v*d_dy(v) - (tau_by_per_rho / h) + Du * (d2_dx2(v) + d2_dy2(v))

        # Update flux
        u = u + du_dt * dt
        v = v + dv_dt * dt

        # Boundary conditions on flux
        # Left boundary
        u[:, :, :, 0] = -u[:, :, :, 1]
        v[:, :, :, 0] = v[:, :, :, 1]

        # Right boundary
        u[:, :, :, -1] = 2*u[:, :, :, -2] - u[:, :, :, -3]
        v[:, :, :, -1] = v[:, :, :, -2]

        # Top
        u[:, :, 0, :] = u[:, :, 1, :]
        v[:, :, 0, :] = -v[:, :, 1, :]

        # Bottom
        u[:, :, -1, :] = u[:, :, -2, :]
        v[:, :, -1, :] = -v[:, :, -2, :]

        # Compute h update
        dh_dt = - d_dx(h*u) - d_dy(h*v) + Hin

        h = h + dh_dt * dt

        # Ensure minimum h to critical value
        h = torch.clamp(h, min=Hc)

        h[:, :, :, 0] = h[:, :, :, 1]
        h[:, :, :, -1] = h[:, :, :, -2]
        h[:, :, 0, :] = h[:, :, 1, :]
        h[:, :, -1, :] = h[:, :, -2, :]

        # Compute sediment update
        Ds = D0 * (1.0 - pD * (b / k))

        topographic_diffusion_term = d_dx(Ds * d_dx(s)) + d_dy(Ds * d_dy(s))

        # effective water height
        he = h - Hc

        ds_dt = Sin * (he / (Qs + he)) - Es * (1.0 - pE * (b/k)) * s * tau_b_per_rho + topographic_diffusion_term

        # Update s
        s = s + ds_dt * dt * 44712

        # Boundary conditions on s
        s[:, :, :, 0] = s[:, :, :, 1]
        s[:, :, :, -1] = 0
        s[:, :, 0, :] = s[:, :, 1, :]
        s[:, :, -1, :] = s[:, :, -2, :]

        # Compute vegetation update
        db_dt = r * b * (1.0 - (b/k)) * (Qq / (Qq + he)) - Eb * b * tau_b_per_rho + Db * (d2_dx2(b) + d2_dy2(b))

        # Update b
        b = b + db_dt * dt * 44712

        # Boundary conditions on b
        b[:, :, :, 0] = b[:, :, :, 1]
        b[:, :, :, -1] = b[:, :, :, -2]
        b[:, :, 0, :] = b[:, :, 1, :]
        b[:, :, -1, :] = b[:, :, -2, :]

        self.h = h
        self.u = u
        self.v = v
        self.s = s
        self.b = b

    def run_iters(self, count):

        # Move everything to gpu
        self.gpu()

        # Run num of iterations
        for _ in range(count):
            self.run_iter()
        
        # Move to cpu
        self.cpu()

    def to(self, device):
        self.h = self.h.to(device)
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.s = self.s.to(device)
        self.b = self.b.to(device)

    def cpu(self):
        self.to(torch.device("cpu"))

    def gpu(self):
        self.to(torch_device)
