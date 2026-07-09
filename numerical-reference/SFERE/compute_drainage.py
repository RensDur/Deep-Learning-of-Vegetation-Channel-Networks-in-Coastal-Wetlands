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

	drainage_report = torch.zeros(num_iterations, 3)

	# Stats
	start_time = time.time()


	# As long as the window is open, run the simulation
	for i in range(num_iterations):

		# Estimate how much water is added by the clamping
		water_added_by_clamp = torch.sum(torch.abs(basin.h[torch.where(basin.h < basin.Hc)]))

		# Make a simulation step
		basin.simulate(dt=sfere_dt)

		# Report the water level
		drainage_report[i, 0] = torch.sum(basin.h)

		drainage_report[i, 1] = water_added_by_clamp
		drainage_report[i, 2] = basin.Hin * basin.width * basin.height

		# As soon as the iteration count is greater than or equal to the next shapshot, we make a snapshot and increment the counter
		if i % 1000 == 0:

			# Every snapshot, make an ETA report
			avg_time_per_iter = (time.time() - start_time)/(i+1)
			iters_to_run = num_iterations - i + 1
			eta = iters_to_run * avg_time_per_iter

			print(f"\r\033[K" + f"SFERE iteration {i}/{num_iterations} - ETA {eta/60:.2f}m - Last drainage rate: {water_added_by_clamp}", end="")
		
	# After completion, store the drainage per iteration
	torch.save(drainage_report.detach().cpu(), "./drainage_report.pt")