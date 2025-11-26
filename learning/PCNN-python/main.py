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

    # Initialize randomization seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize torch CPU threads
    torch.set_num_threads(8)

    # Extract parameters
    params = parameters.params()
    print(f"Parameters: {vars(params)}")

    # Create dataset
    data = dataset.Dataset(params, torch_device)

    # Create solver
    solver = pcnn_solver.PCNNSolver(data, params, torch_device)

    solver.train()





if __name__ == "__main__":
    main()