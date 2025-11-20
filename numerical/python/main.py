from numerical_solver import NumericalSolver
import matplotlib.pyplot as plt
import threading
import numpy as np

solver = NumericalSolver()


# Plot domain (first time)
plt.ion()

# Create subplots
figure, axs = plt.subplots(2, 2, figsize=(20, 10))

sediment_plot = axs[0, 0].imshow(solver.domain.get_S(), cmap="gray", vmin=0, vmax=0.2)
sediment_plot_under_veg = axs[0, 1].imshow(solver.domain.get_S(), cmap="gray", vmin=0, vmax=0.2)
vegetation_plot = axs[0, 1].imshow(solver.domain.get_B(), cmap="YlGn", vmin=0, vmax=solver.domain.k, alpha=0.8)

momentum_u_plot = axs[1, 0].imshow(solver.domain.get_u(), cmap="bwr", vmin=-2, vmax=2)
momentum_v_plot = axs[1, 1].imshow(solver.domain.get_v(), cmap="bwr", vmin=-2, vmax=2)

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

    while running:
        for _ in range(200):
            solver.solve_step()

        print(f"Simulated until t={solver.t} M2-tidal periods")
        print(f"Maximum u encountered = {np.max(solver.domain.get_u())}")

        sediment_plot.set_data(solver.domain.get_S())
        sediment_plot_under_veg.set_data(solver.domain.get_S())
        vegetation_plot.set_data(solver.domain.get_B())

        momentum_u_plot.set_data(solver.domain.get_u())
        momentum_v_plot.set_data(solver.domain.get_v())



if __name__ == "__main__":
    # Start the simulation thread
    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.start()

    # Run the main thread to open the plot-window
    main()

    # Join the sim-thread
    sim_thread.join()