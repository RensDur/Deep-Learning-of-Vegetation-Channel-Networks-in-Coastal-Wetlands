import multiprocessing
import torch
import numpy as np
import parameters
from dataset import Dataset
from spline_pinn_solver import SplinePINNSolver

# Find the number of available CPUs, capped at 8
NUM_CPUS = min(multiprocessing.cpu_count(), 8)
torch.set_num_threads(NUM_CPUS)
print(f"Using {NUM_CPUS} threads")

# Find the GPU device for pytorch
torch_device = torch.device('cpu')  # Default to CPU
# Switch to MPS (Apple Metal) if available
if torch.backends.mps.is_available():
    torch_device = torch.device('mps')
# Or CUDA if we're on an Nvidia machine
elif torch.cuda.is_available():
    torch_device = torch.device('cuda')
print(f"Using torch device '{torch_device}'")

# Initialize randomization seeds
torch.manual_seed(0)
np.random.seed(0)



def main():

    # Extract parameters
    params = parameters.params()
    print(f"Parameters: {vars(params)}")

    # Create dataset
    dataset = Dataset(params, torch_device)

    print(f"Dataset hidden state size: {dataset.variables.hidden_size()}")

    # Create solver
    solver = SplinePINNSolver(dataset, params, torch_device)

    # Run training routine
    solver.train()


if __name__ == '__main__':
    main()