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

dt = 0.001

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
    return F.conv2d(quantity, dx_kernel, padding=(0,1))

def d_dy(quantity):
    return F.conv2d(quantity, dy_kernel, padding=(1,0))

def d2_dx2(quantity):
    return F.conv2d(quantity, dx2_kernel, padding=(0,1))

def d2_dy2(quantity):
    return F.conv2d(quantity, dy2_kernel, padding=(1,0))


class Solver():


    def __init__(self):
        #
        # ENVIRONMENTS
        #

        self.width = 200
        self.height = 200

        self.h = torch.zeros(1, 1, self.height, self.width)
        self.hu = torch.zeros(1, 1, self.height, self.width)
        self.hv = torch.zeros(1, 1, self.height, self.width)
        self.s = torch.zeros(1, 1, self.height, self.width)
        self.b = torch.zeros(1, 1, self.height, self.width)

        # H: initial condition
        self.h[:, :, :, :] = H0

        # Randomly place vegetation
        self.b[torch.where(torch.rand(1, 1, self.height, self.width) < 0.002)] = k

    #
    # MAIN FUNCTION
    #
    def run_iter(self):

        h = self.h
        hu = self.hu
        hv = self.hv
        s = self.s
        b = self.b

        # Compute u and v
        u = hu / h
        v = hv / h

        # Manning's n
        n = nb + (nv - nb) * (b / k)

        # Chezys coefficient
        Cz = (1.0 / n) * torch.pow(h, 1.0 / 6.0)

        tau_precalc = (grav / torch.pow(Cz, 2.0)) * torch.pow(torch.pow(u, 2) + torch.pow(v, 2), 0.5)

        tau_bx_per_rho = tau_precalc * u
        tau_by_per_rho = tau_precalc * v
        tau_b_per_rho = (grav / torch.pow(Cz, 2.0)) * (torch.pow(u, 2) + torch.pow(v, 2))

        # Compute flux update
        dhu_dt = -grav*h*d_dx(s + h) - hu*(d_dx(u) + d_dy(v)) - u*d_dx(hu) - v*d_dy(hu) - tau_bx_per_rho
        dhv_dt = -grav*h*d_dy(s + h) - hv*(d_dx(u) + d_dy(v)) - u*d_dx(hv) - v*d_dy(hv) - tau_by_per_rho

        # Update flux
        hu = hu + dhu_dt * dt
        hv = hv + dhv_dt * dt

        # Boundary conditions on flux
        # Left boundary
        hu[:, :, :, 0] = -hu[:, :, :, 1]
        hv[:, :, :, 0] = hv[:, :, :, 1]

        # Right boundary
        hu[:, :, :, -1] = 2*hu[:, :, :, -2] - hu[:, :, :, -3]
        hv[:, :, :, -1] = hv[:, :, :, -2]

        # Top
        hu[:, :, 0, :] = hu[:, :, 1, :]
        hv[:, :, 0, :] = -hv[:, :, 1, :]

        # Bottom
        hu[:, :, -1, :] = hu[:, :, -2, :]
        hv[:, :, -1, :] = -hv[:, :, -2, :]

        # Compute h update
        dh_dt = - d_dx(hu) - d_dy(hv) + Hin

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
        self.hu = hu
        self.hv = hv
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
        self.hu = self.hu.to(device)
        self.hv = self.hv.to(device)
        self.s = self.s.to(device)
        self.b = self.b.to(device)

    def cpu(self):
        self.to(torch.device("cpu"))

    def gpu(self):
        self.to(torch_device)
