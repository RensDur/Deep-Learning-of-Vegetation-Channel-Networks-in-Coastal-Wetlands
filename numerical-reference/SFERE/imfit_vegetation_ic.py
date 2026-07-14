import torch
import time
from window import MultiWindow
from sfere import SaltmarshDomain


if __name__ == "__main__":

    window = MultiWindow(800, 800)

    basin = SaltmarshDomain(800, 800, torch.device("mps"))
    basin.dx = 1/4
    basin.dy = 1/4

    # Load the vegetation initial condition that has been pulled through the ImFit-CNN
    basin.b = torch.load(f"./imfit_vegetation_ic.pt", map_location=basin.device)

    # Open the window
    window.open()

    iteration_count = 0
    start_time = time.time()

    # As long as the window is open, run the simulation
    while window.is_open and iteration_count < 10_000_000:

        # Make a simulation step
        basin.simulate(dt=0.001)

        if iteration_count % 10_000 == 0:
            # Update the window state
            window.set_data(
                basin.h[0, 0],
                basin.u[0, 0],
                basin.v[0, 0],
                basin.s[0, 0],
                basin.b[0, 0]
            )

            window.update()

            process_time = time.time() - start_time
            time_per_iteration = process_time / (iteration_count + 1)
            eta = (10_000_000 - iteration_count) * time_per_iteration

            print("\r" + (" " * 100), end="")
            print(f"\rCompleted {iteration_count} iterations in {process_time:.2f}s\tETA {eta/60:.2f} minutes", end="")

        iteration_count += 1

    # After simulation is complete (until 1M iterations), simply continuously update the window
    while window.is_open:
        window.update()
