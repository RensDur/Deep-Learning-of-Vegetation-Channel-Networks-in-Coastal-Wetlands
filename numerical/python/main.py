from numerical_solver import NumericalSolver
import matplotlib.pyplot as plt


solver = NumericalSolver()


# Plot domain (first time)
plt.ion()

# Create subplots
figure, (axLeft, axRight) = plt.subplots(1, 2, figsize=(20, 10))
water_thickness_plot = axLeft.imshow(solver.domain.get_h(), cmap="Blues", vmin=0.95, vmax=1.1)
momentum_plot = axRight.imshow(solver.domain.get_h(), cmap="Greens", vmin=-0.5, vmax=0.5)

plt.colorbar(water_thickness_plot)
plt.colorbar(momentum_plot)

# setting title
axLeft.set(title="Water layer thickness")
axRight.set(title="U Momentum")

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

    water_thickness_plot.set_data(solver.domain.get_h())
    momentum_plot.set_data(solver.domain.get_u())

    # Plot the domain (update existing plot)
    # Draw updated values
    figure.canvas.draw()

    # UI Loop: process all pending UI events
    figure.canvas.flush_events()
