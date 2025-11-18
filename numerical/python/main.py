from numerical_solver import NumericalSolver
import matplotlib.pyplot as plt
import time

solver = NumericalSolver()


# Plot domain (first time)
plt.ion()

# Create subplots
figure, ax = plt.subplots(figsize=(10, 10))
plot_image = ax.imshow(solver.domain.get_h())

# setting title
plt.title("Water layer thickness", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("Along shore")
plt.ylabel("Cross shore")

running = True
while running:

    for _ in range(100):
        solver.solve_step()

    plot_image.set_data(solver.domain.get_h())

    # print(solver.domain.get_h())

    # Plot the domain (update existing plot)
    # Draw updated values
    figure.canvas.draw()

    # UI Loop: process all pending UI events
    figure.canvas.flush_events()
