import multiprocessing
import torch
import numpy as np
import parameters
import dataset
import pcnn_solver

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

    # Because we're visualizing, create only one domain
    params.dataset_size = 1
    params.batch_size = 1

    print(f"Parameters: {vars(params)}")

    # Create dataset
    data = dataset.Dataset(params, torch_device)

    # Create solver
    solver = pcnn_solver.PCNNSolver(data, params, torch_device)

    # Visualize the output
    solver.visualize_numerical()


if __name__ == "__main__":
    main()