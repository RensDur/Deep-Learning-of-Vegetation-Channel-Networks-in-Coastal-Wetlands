import torch
import numpy as np
from window import Window
from numerical import Solver


def main():

    # Initialize randomization seeds
    torch.manual_seed(1)
    np.random.seed(6)

    width = 100
    height = 100

    torch_device = torch.device("cpu")

    if torch.backends.mps.is_available():
        torch_device = torch.device("mps")

    print(f"Using torch device: {torch_device}")
    
    win = Window("h", width, height)
    win.set_data_range(-0.1, 0.1)

    solver = Solver(width, height, torch_device)

    #
    # Simulation loop
    #

    with torch.no_grad():
        while win.is_open():

            # Run a simulation step
            for _ in range(100):
                solver.step()

            # Show the result
            h, hu, hv, s = solver.get()

            win.put_image(hu[0, 0])
            win.update()



if __name__ == "__main__":
    main()