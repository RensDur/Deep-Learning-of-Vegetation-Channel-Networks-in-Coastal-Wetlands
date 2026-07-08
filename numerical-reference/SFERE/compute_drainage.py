import os
import time
import torch
from sfere import SaltmarshDomain



def compute_log_slowdown_curve(steepness, max_iter, total_samples):
	xs = torch.linspace(0, steepness, total_samples)
	ys = torch.exp(xs)
	ys = ys - torch.min(ys)
	ys = ys / torch.max(ys) * max_iter
	xs = xs / steepness * total_samples

	return xs, ys


if __name__ == "__main__":

	basin = SaltmarshDomain(800, 800, torch.device("mps"))
	basin.dx = 1/4
	basin.dy = 1/4

	# Set a timestep (identical to the one used in SFERE, but necessary explicitly for clarity of this script)
	sfere_dt = 0.0125

	num_iterations = 1_000_000

	drainage_per_iteration = torch.zeros(num_iterations)

	# Stats
	start_time = time.time()


	# As long as the window is open, run the simulation
	for i in range(num_iterations):

		# Compute total water volume BEFORE
		water_volume_before = torch.sum(basin.h)

		# Make a simulation step
		basin.simulate(dt=sfere_dt)

		# Compute total water volume AFTER
		water_volume_after = torch.sum(basin.h)

		# Drainage equals difference between water volume BEFORE and AFTER (per timestep)
		drainage = (water_volume_after - water_volume_before)/sfere_dt

		drainage_per_iteration[i] = drainage

		# As soon as the iteration count is greater than or equal to the next shapshot, we make a snapshot and increment the counter
		if i % 1000 == 0:

			# Every snapshot, make an ETA report
			avg_time_per_iter = (time.time() - start_time)/(i+1)
			iters_to_run = num_iterations - i + 1
			eta = iters_to_run * avg_time_per_iter

			print(f"\r\033[K" + f"SFERE iteration {i}/{num_iterations} - ETA {eta/60:.2f}m - Last drainage rate: {drainage}", end="")
		
	# After completion, store the drainage per iteration
	torch.save(drainage_per_iteration.detach().cpu(), "./drainage_per_iteration.pt")