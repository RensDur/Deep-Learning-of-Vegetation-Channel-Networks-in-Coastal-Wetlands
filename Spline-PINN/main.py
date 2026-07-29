import multiprocessing
import torch
import numpy as np
import parameters
import dataset
import spline_pinn_solver
from window import MultiWindow
from videoplayer import VideoChannel, VideoPlayer

def main():

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
    torch.manual_seed(5)
    np.random.seed(5)

    # Because we're visualizing, create only one domain
    params.dataset_size = 1
    params.batch_size = 1

    print(f"Parameters: {vars(params)}")

    # Create dataset
    data = dataset.Dataset(params, torch_device, types=["oscillator"])

    # print(data.env_info[0])

    # Create solver
    solver = spline_pinn_solver.SplinePINNSolver(data, params, torch_device)

    # Create window
    win = MultiWindow(params.width * params.resolution_factor, params.height * params.resolution_factor)

    # Create video player
    video_channels = [
        VideoChannel("h", (-0.2, 0.2), "Blues"),
        VideoChannel("u", (-1, 1), "bwr"),
        VideoChannel("v", (-1, 1), "bwr")
    ]

    video_player = VideoPlayer(params.width * params.resolution_factor, params.height * params.resolution_factor, video_channels, 60, 24)

    # Visualize the output
    solver.visualize(video_player)


if __name__ == "__main__":
    main()