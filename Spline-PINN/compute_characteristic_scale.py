import torch
import torch.nn.functional as F
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

# Default parameters
class Parameters():

    def __init__(self):
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




def main():

    # Find the number of available CPUs, capped at 8
    NUM_CPUS = min(multiprocessing.cpu_count(), 8)
    torch.set_num_threads(NUM_CPUS)
    print(f"Using {NUM_CPUS} threads")

    # Select a torch device
    torch_device = torch.device('cpu')  # Default to CPU
    # Switch to MPS (Apple Metal) if available
    if torch.backends.mps.is_available():
        torch_device = torch.device('mps')
    # Or CUDA if we're on an Nvidia machine
    elif torch.cuda.is_available():
        torch_device = torch.device('cuda')
    print(f"Using torch device '{torch_device}'")

    #
    # Convolution kernels
    #

    dx = 1/4
    dy = 1/4

    # First order derivatives
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
        return F.conv2d(quantity, dx2_kernel, padding=(0,1)) / (dx**2)

    def d2_dy2(quantity):
        return F.conv2d(quantity, dy2_kernel, padding=(1,0)) / (dy**2)

    # Load all sfere samples (validation set) from disk
    sfere_outputs = torch.zeros(500, 5, 800, 800)

    for i in range(500):
        sfere_outputs[i] = torch.cat([
            torch.load(f"./snapshots-log-slowdown/snapshot_{i}/h.pt", map_location=torch.device("cpu")),
            torch.load(f"./snapshots-log-slowdown/snapshot_{i}/u.pt", map_location=torch.device("cpu")),
            torch.load(f"./snapshots-log-slowdown/snapshot_{i}/v.pt", map_location=torch.device("cpu")),
            torch.load(f"./snapshots-log-slowdown/snapshot_{i}/s.pt", map_location=torch.device("cpu")),
            torch.load(f"./snapshots-log-slowdown/snapshot_{i}/b.pt", map_location=torch.device("cpu"))
        ], dim=1)

    print(f"Loaded all items from disk; computing characteristic scale for each sample")

    # Characteristic scales per variable h, u, v, BC-closed and BC-open
    characteristic_scales = torch.zeros(500, 5)

    # Load parameters
    params = Parameters()

    # Go over each sample
    for i in tqdm(range(500)):

        h = sfere_outputs[i:(i+1), 0:1].to(torch_device)
        u = sfere_outputs[i:(i+1), 1:2].to(torch_device)
        v = sfere_outputs[i:(i+1), 2:3].to(torch_device)
        s = sfere_outputs[i:(i+1), 3:4].to(torch_device)
        b = sfere_outputs[i:(i+1), 4:5].to(torch_device)

        # Characteristic scale of h
        h_scale_local = torch.abs(h * d_dx(u) + u * d_dx(h)) + torch.abs(h * d_dy(v) + v * d_dy(h))

        # Compute 90th percentile
        h_scale_90th_percentile = torch.quantile(h_scale_local.flatten(), 0.9)
        characteristic_scales[i, 0] = h_scale_90th_percentile

        # Characteristic scale of momentum terms
        n = params.nb + (params.nv - params.nb) * (b / params.k)
        chezy = (1.0 / n) * torch.pow(h, 1/6)
        tau_precalc = (params.grav / torch.pow(chezy, 2)) * torch.pow(torch.pow(u, 2) + torch.pow(v, 2), 0.5)
        tau_bx_per_rho = tau_precalc * u
        tau_by_per_rho = tau_precalc * v

        u_scale_local = torch.abs(d_dx(h)) + torch.abs(d_dx(s)) + torch.abs(u * d_dx(u)) + torch.abs(v * d_dy(u)) + torch.abs(tau_bx_per_rho / h) + torch.abs(d2_dx2(u) + d2_dy2(u))
        v_scale_local = torch.abs(d_dy(h)) + torch.abs(d_dy(s)) + torch.abs(u * d_dx(v)) + torch.abs(v * d_dy(v)) + torch.abs(tau_by_per_rho / h) + torch.abs(d2_dx2(v) + d2_dy2(v))

        # Compute 90th percentile
        u_scale_90th_percentile = torch.quantile(u_scale_local.flatten(), 0.9)
        v_scale_90th_percentile = torch.quantile(v_scale_local.flatten(), 0.9)

        characteristic_scales[i, 1] = u_scale_90th_percentile
        characteristic_scales[i, 2] = v_scale_90th_percentile

        # Compute boundary term magnitude [CLOSED | OPEN] over boundary band of width 5%
        boundary_band_width = int(800 * 0.05)

        bc_closed_scale_local = torch.abs(d_dx(h)) + torch.abs(d_dy(h)) + torch.abs(u) + torch.abs(v)
        bc_closed_scale_local[0, 0, boundary_band_width:, boundary_band_width:-boundary_band_width] = 0 # Remove everything that's not included in the closed boundary band

        bc_open_scale_local = torch.abs(d_dx(h)) + torch.abs(d_dy(h))
        bc_open_scale_local[0, 0, :-boundary_band_width, :] = 0 # Remove everything that's not included in the open boundary band

        # Compute 90th percentile
        bc_closed_90th_percentile = torch.quantile(bc_closed_scale_local.flatten(), 0.9)
        bc_open_90th_percentile = torch.quantile(bc_open_scale_local.flatten(), 0.9)

        characteristic_scales[i, 3] = bc_closed_90th_percentile
        characteristic_scales[i, 4] = bc_open_90th_percentile


    # Store the characteristic scales per sample and per variable to disk
    torch.save(characteristic_scales.detach().cpu(), f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt")



    plt.plot(characteristic_scales[:, 0])
    plt.plot(characteristic_scales[:, 1])
    plt.plot(characteristic_scales[:, 2])
    plt.plot(characteristic_scales[:, 3])
    plt.plot(characteristic_scales[:, 4])
    plt.show()
    


if __name__ == "__main__":
    main()