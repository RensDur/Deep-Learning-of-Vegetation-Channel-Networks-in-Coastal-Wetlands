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

	xs, ys = compute_log_slowdown_curve(3.3, 1_000_000, 500)

	assert xs.shape[0] == 500
	assert xs.shape == ys.shape

	# Create an output folder
	output_folder = f"snapshots-log-slowdown"
	os.makedirs(f"./{output_folder}", exist_ok=True)

	# Stats
	snapshots_taken = 0
	snapshot_idx = 0
	sfere_iterations = 0

	start_time = time.time()


	# As long as the window is open, run the simulation
	while snapshot_idx < xs.shape[0]:

		# Make a simulation step
		basin.simulate()
		sfere_iterations += 1

		# As soon as the iteration count is greater than or equal to the next shapshot, we make a snapshot and increment the counter
		if sfere_iterations >= ys[snapshot_idx]:

			# Create a snapshot
			os.makedirs(f"./{output_folder}/{sfere_iterations}", exist_ok=True)

			torch.save(basin.h.detach().cpu(), f"./{output_folder}/{sfere_iterations}/h.pt")
			torch.save(basin.u.detach().cpu(), f"./{output_folder}/{sfere_iterations}/u.pt")
			torch.save(basin.v.detach().cpu(), f"./{output_folder}/{sfere_iterations}/v.pt")
			torch.save(basin.s.detach().cpu(), f"./{output_folder}/{sfere_iterations}/s.pt")
			torch.save(basin.b.detach().cpu(), f"./{output_folder}/{sfere_iterations}/b.pt")

			# Increment the counter
			snapshot_idx += 1

		# Every 1000 iterations, make an ETA report
		if sfere_iterations % 1000 == 0:

			avg_time_per_iter = (time.time() - start_time)/sfere_iterations
			iters_to_run = ys[-1] - sfere_iterations
			eta = iters_to_run * avg_time_per_iter

			iters_to_next_shapshot = ys[snapshot_idx] - sfere_iterations
			eta_until_next_snapshot = iters_to_next_shapshot * avg_time_per_iter

			print(f"\rSFERE iteration {sfere_iterations} - Taken {snapshot_idx} snapshots - Iterations until next shapshot: {iters_to_next_shapshot} ({eta_until_next_snapshot/60:.2f}m) - ETA {eta/60:.2f}m", end="")
		
