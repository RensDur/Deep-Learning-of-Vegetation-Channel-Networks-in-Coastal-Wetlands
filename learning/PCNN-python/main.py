import torch
import numpy as np
import parameters
import dataset
import pcnn_solver


def main():


    # Select torch device
    torch_device = torch.device('cpu')  # Default to CPU
    # Switch to MPS (Apple Metal) if available
    if torch.backends.mps.is_available():
        torch_device = torch.device('mps')
    # Or CUDA if we're on an Nvidia machine
    elif torch.cuda.is_available():
        torch_device = torch.device('cuda')

    print(f"Using torch device '{torch_device}'")

    # Initialize torch CPU threads
    torch.set_num_threads(8)

    # Extract parameters
    params = parameters.params()

    # If visualization is requested through parameter '--visualize'
    if params.visualize:
        params.dataset_size = 1
        params.batch_size = 1

    print(f"Parameters: {vars(params)}")

    # Create dataset
    data = dataset.Dataset(params, torch_device)

    # Create solver
    solver = pcnn_solver.PCNNSolver(data, params, torch_device)

    if params.visualize:
        solver.visualize()
    else:
        solver.train()





if __name__ == "__main__":
    main()