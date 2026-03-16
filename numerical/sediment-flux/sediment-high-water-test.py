import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from window import Window

#
# TORCH DEVICE
#

torch_device = torch.device("cpu")

if torch.backends.mps.is_available():
    torch_device = torch.device("mps")

print(f"Using torch device: {torch_device}")


#
# ENVIRONMENTS
#

width = 100
height = 100

h = torch.ones(1, 1, height, width, device=torch_device)
hu = torch.zeros(1, 1, height, width, device=torch_device)
hv = torch.zeros(1, 1, height, width, device=torch_device)
s = torch.zeros(1, 1, height, width, device=torch_device)

#
# PARAMETERS
#

H0 = 1
Hc = 1e-3
grav = 9.81

D0 = 10e-7
Sin = 5e-9
Qs = 6e-4
Es = 2.5e-4
nb = 0.016

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

#
# MAIN WINDOW
#
win = Window("Numerical Simulation", width, height)



win.set_data_range(0, 0.2)

# MAIN LOOP
with torch.no_grad():
    # Simulation loop
    while win.is_open():

        # Compute u and v
        u = hu / h
        v = hv / h

        # Manning's n
        n = nb # No vegetation

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
        hu[:, :, :, -1] = -hu[:, :, :, -2]
        hv[:, :, :, -1] = hv[:, :, :, -2]

        # Top
        hu[:, :, 0, :] = hu[:, :, 1, :]
        hv[:, :, 0, :] = -hv[:, :, 1, :]

        # Bottom
        hu[:, :, -1, :] = hu[:, :, -2, :]
        hv[:, :, -1, :] = -hv[:, :, -2, :]

        # Compute h update
        dh_dt = - d_dx(hu) - d_dy(hv)

        h = h + dh_dt * dt

        h[:, :, :, 0] = h[:, :, :, 1]
        h[:, :, :, -1] = h[:, :, :, -2]
        h[:, :, 0, :] = h[:, :, 1, :]
        h[:, :, -1, :] = h[:, :, -2, :]

        # Oscillator
        # h[:, :, (height//2-5):(height//2+5), (width//2-5):(width//2+5)] = 1 + 0.5 * torch.sin(torch.Tensor([current_t])).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 10, 10)

        # Compute sediment update

        Ds = D0 # No vegetation

        # effective water height
        he = h - Hc

        ds_dt = Sin * (he / (Qs + he)) - Es * s * tau_b_per_rho + Ds * (d2_dx2(s) + d2_dy2(s))

        # Update s
        s = s + ds_dt * dt * 44712

        # Boundary conditions on s
        s[:, :, :, -10:] = 0

        current_t += dt

        # Update view
        win.put_image(s[0, 0].cpu())
        win.update()
