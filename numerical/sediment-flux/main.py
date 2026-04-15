import os
import datetime
from simulation_logic import Solver
import matplotlib.pyplot as plt
import threading
import torch
import numpy as np

solver = Solver()

# Plot domain (first time)
plt.ion()

# Create subplots
figure, axs = plt.subplots(2, 2, figsize=(20, 10))

sediment_plot = axs[0, 0].imshow(solver.s[0,0], cmap="gray", vmin=0, vmax=0.2)
sediment_plot_under_veg = axs[0, 1].imshow(solver.s[0,0], cmap="gray", vmin=0, vmax=0.2)
vegetation_plot = axs[0, 1].imshow(solver.b[0,0], cmap="YlGn", vmin=0, vmax=1500, alpha=0.8)

momentum_u_plot = axs[1, 0].imshow(solver.u[0,0], cmap="bwr", vmin=-0.2, vmax=0.2)
momentum_v_plot = axs[1, 1].imshow(solver.v[0,0], cmap="bwr", vmin=-0.2, vmax=0.2)

# setting title
axs[0, 0].set(title="Sediment bed", xlabel="Cross shore", ylabel="Along shore")
axs[0, 1].set(title="Sediment bed with vegetation", xlabel="Cross shore", ylabel="Along shore")
axs[1, 0].set(title="Momentum u (x-direction)", xlabel="Cross shore", ylabel="Along shore")
axs[1, 1].set(title="Momentum v (y-direction)", xlabel="Cross shore", ylabel="Along shore")

# Color bars
plt.colorbar(sediment_plot)
plt.colorbar(sediment_plot_under_veg)
plt.colorbar(vegetation_plot)
plt.colorbar(momentum_u_plot)
plt.colorbar(momentum_v_plot)

# In interactive mode, plt.show() immediately returns
plt.show()

# Store intermediate results after several number of iterations
store_points = [i for i in range(10_000, 2_000_000, 10_000)]

# Let the program run until the 'closing event' has been fired
running = True

def __on_figure_close(event):
    global running
    running = False

def main():
    global running
    global figure

    figure.canvas.mpl_connect('close_event', __on_figure_close)

    while running:

        # Plot the domain (update existing plot)
        # Draw updated values
        figure.canvas.draw()

        # UI Loop: process all pending UI events
        figure.canvas.flush_events()

def simulation_loop():
    global running
    global solver
    global sediment_plot
    global sediment_plot_under_veg
    global vegetation_plot
    global momentum_u_plot
    global momentum_v_plot

    # Create the storage directory
    date_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(f"out",exist_ok=True)

    iter_sum = 0
    n_iter = 1000

    while running:
        
        try:
            print(f"Iter {iter_sum}")
            
            solver.run_iters(n_iter)
            iter_sum += n_iter

            sediment_plot.set_data(solver.s[0,0])
            sediment_plot_under_veg.set_data(solver.s[0,0])
            vegetation_plot.set_data(solver.b[0,0])

            momentum_u_plot.set_data(solver.u[0,0])
            momentum_v_plot.set_data(solver.v[0,0])

        except:
            running = False

        

        # Store the result if a storage-point has been reached
        if len(store_points) > 0:
            if iter_sum >= store_points[0]:

                print(f"Writing iteration {store_points[0]} to disk")

                # Store point reached, store a snapshot
                os.makedirs(f"out/{store_points[0]}",exist_ok=True)

                torch.save(solver.h, f"out/{store_points[0]}/h.pt")
                torch.save(solver.u, f"out/{store_points[0]}/u.pt")
                torch.save(solver.v, f"out/{store_points[0]}/v.pt")
                torch.save(solver.s, f"out/{store_points[0]}/s.pt")
                torch.save(solver.b, f"out/{store_points[0]}/b.pt")

                # Onto next store_point
                del store_points[0]

        # Once all requested store-points have passed, stop the simulation
        if len(store_points) == 0:
            running = False



if __name__ == "__main__":
    # Start the simulation thread
    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.start()

    # Run the main thread to open the plot-window
    main()

    # Join the sim-thread
    sim_thread.join()