import torch
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from spline_pinn_solver import SplinePINNSolver
from numerical_solver import NumericalSolver
import parameters
import matplotlib.pyplot as plt
import multiprocessing




def main():

    # For diagnostics, create a database with one environment of the selected type

    # Extract parameters
    params = parameters.params()

    # Find the number of available CPUs, capped at 8
    NUM_CPUS = min(multiprocessing.cpu_count(), 8)
    torch.set_num_threads(NUM_CPUS)
    print(f"Using {NUM_CPUS} threads")

    # Find the GPU device for pytorch
    torch_device = torch.device('cpu')  # Default to CPU
    # Switch to MPS (Apple Metal) if available
    if torch.backends.mps.is_available() and params.cuda:
        torch_device = torch.device('mps')
    # Or CUDA if we're on an Nvidia machine
    elif torch.cuda.is_available() and params.cuda:
        torch_device = torch.device('cuda')
    print(f"Using torch device '{torch_device}'")

    # Initialize randomization seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Because we're visualizing, create only one domain
    params.dataset_size = 1
    params.batch_size = 1

    print(f"Parameters: {vars(params)}")

    # Create a dataset
    dataset = Dataset(params, torch_device, types=["oscillator"])

    # Create two solvers: Spline PINN and a Numerical reference
    pinn_solver = SplinePINNSolver(dataset, params, torch_device)
    num_solver = NumericalSolver(params, torch_device)

    # Place the pinn solver in EVAL mode
    # This will load the correct model from disk and place it in evaluation mode
    pinn_solver.eval_mode()

    # We let the numerical simulator make a configurable number of steps to reach the same timestep as the pinn (params.dt)
    num_resolution = 10
    num_timestep = params.dt / num_resolution

    #
    # MAIN LOOP
    #

    while True:

        # Perform one iteration of 'params.dt' using the pinn
        # This brings the system from 'old_hidden_state' to 'new_hidden_state' with timstep 'params.dt'
        old_hidden_state, new_hidden_state, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask = pinn_solver.step()

        # Obtain the starting point from the old_hidden_state, at the selected collocation points
        h, grad_h, hu, grad_hu, hv, grad_hv, s, grad_s, laplacian_s = dataset.interpolate_superres(old_hidden_state, params.resolution_factor)

        # Start at t=0
        for i in range(num_resolution):

            # Perform a numerical step
            num_dh_dt, num_dh_dtu, num_dh_dtv, num_ds_dt = num_solver.step(h, grad_h, hu, grad_hu, hv, grad_hv, s, grad_s, laplacian_s, num_timestep, h_in, h_cond, h_mask, uv_cond, uv_mask, s_cond, s_mask)

            # Compute the spline pinn equivalent
            # And also automatically update h, grad_h, hu, grad_hu, hv, grad_hv, s, grad_s, laplacian_s to the next timestep
            h, grad_h, dh_dt, hu, grad_hu, dhu_dt, hv, grad_hv, dhv_dt, s, grad_s, laplacian_s, ds_dt = dataset.interpolate_superres_timestep(old_hidden_state, new_hidden_state, params.resolution_factor, (i+1)*num_timestep)

            # Compute the error
            mse_h = torch.mean(torch.pow(dh_dt - num_dh_dt, 2.0))
            mse_hu = torch.mean(torch.pow(dhu_dt - num_dhu_dt, 2.0))
            mse_hv = torch.mean(torch.pow(dhv_dt - num_dhv_dt, 2.0))
            mse_s = torch.mean(torch.pow(ds_dt - num_ds_dt, 2.0))

            




    







if __name__ == "__main__":
    main()