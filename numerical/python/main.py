from numerical_solver import NumericalSolver
import matplotlib.pyplot as plt

solver = NumericalSolver()


# Plot domain (first time)
plt.ion()

# Create subplots
figure, ax = plt.subplots(figsize=(10, 10))
ax.imshow(solver.domain.numpy())

# setting title
plt.title("Water layer thickness", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("Along shore")
plt.ylabel("Cross shore")

running = True
while running:

    # Plot the domain (update existing plot)
    # Draw updated values
    figure.canvas.draw()

    # UI Loop: process all pending UI events
    figure.canvas.flush_events()