from numerical_solver import NumericalSolver
import matplotlib.pyplot as plt


solver = NumericalSolver()


# Plot domain (first time)
plt.ion()

# Create subplots
figure, (axLeft, axRight) = plt.subplots(1, 2, figsize=(20, 10))

sediment_plot_left = axLeft.imshow(solver.domain.get_S(), cmap="gray", vmin=0, vmax=0.2)
sediment_plot_right = axRight.imshow(solver.domain.get_S(), cmap="gray", vmin=0, vmax=0.2)
vegetation_plot_right = axRight.imshow(solver.domain.get_B(), cmap="YlGn", vmin=0, vmax=solver.domain.k, alpha=0.8)

# setting title
axLeft.set(title="Sediment bed")
axRight.set(title="Sediment bed with vegetation")

# setting x-axis label and y-axis label
axLeft.set(xlabel="Along shore", ylabel="Cross shore")
axRight.set(xlabel="Along shore", ylabel="Cross shore")

# In interactive mode, plt.show() immediately returns
plt.show()

# Let the program run until the 'closing event' has been fired
running = True

def __on_figure_close(event):
    global running
    running = False

figure.canvas.mpl_connect('close_event', __on_figure_close)

while running:

    for _ in range(100):
        solver.solve_step()

    sediment_plot_left.set_data(solver.domain.get_S())
    sediment_plot_right.set_data(solver.domain.get_S())
    vegetation_plot_right.set_data(solver.domain.get_B())

    # Plot the domain (update existing plot)
    # Draw updated values
    figure.canvas.draw()

    # UI Loop: process all pending UI events
    figure.canvas.flush_events()
