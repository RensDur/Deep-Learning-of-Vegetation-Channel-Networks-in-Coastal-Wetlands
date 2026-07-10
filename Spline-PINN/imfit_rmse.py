import torch
import math
import matplotlib.pyplot as plt
from imfit_general import CompoundFitNet, FitDataset
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray


def root_mean_square_error(img1, img2):
	img1_mean = torch.mean(img1)

	numerator = torch.sum(torch.pow(img1 - img2, 2))
	denominator = torch.sum(torch.pow(img1 - img1_mean, 2))


	return numerator / denominator


def main():

	# Select a torch device
	torch_device = torch.device('cpu')  # Default to CPU
	# Switch to MPS (Apple Metal) if available
	if torch.backends.mps.is_available():
		torch_device = torch.device('mps')
	# Or CUDA if we're on an Nvidia machine
	elif torch.cuda.is_available():
		torch_device = torch.device('cuda')
	print(f"Using torch device '{torch_device}'")

	# Load all sfere samples from disk
	sfere_outputs = torch.zeros(500, 5, 800, 800)

	for i in range(500):
		sfere_outputs[i] = torch.cat([
				torch.load(f"./snapshots-log-slowdown/snapshot_{i}/h.pt", map_location=torch.device("cpu")),
				torch.load(f"./snapshots-log-slowdown/snapshot_{i}/u.pt", map_location=torch.device("cpu")),
				torch.load(f"./snapshots-log-slowdown/snapshot_{i}/v.pt", map_location=torch.device("cpu")),
				torch.load(f"./snapshots-log-slowdown/snapshot_{i}/s.pt", map_location=torch.device("cpu")),
				torch.load(f"./snapshots-log-slowdown/snapshot_{i}/b.pt", map_location=torch.device("cpu"))
		], dim=1)

	# Load the training dataset for the imfit net
	training_dataset = FitDataset(800, 800, torch_device)

	# Load the imfit net
	imfit_net = CompoundFitNet(training_dataset.variables, torch_device)
	imfit_net.load_state_from(f"imfit_output/{training_dataset.variables.summary()}") # Immediately load the pre-trained state from disk
	imfit_net.eval()

	# Process the images in batches of 50
	batch_size = 1
	processed = 0
	imfit_outputs = torch.zeros(500, 5, 800, 800)

	while processed < 500:
		this_batch = min(batch_size, 500 - processed)
		batch_slice = slice(processed, processed + this_batch)

		# Process all images through the imfit net
		batch_output = imfit_net(sfere_outputs[batch_slice].to(torch_device)).detach()

		# Sample all imfit outputs at the same 800x800 resolution as the input images
		h, grad_h, u, grad_u, v, grad_v, s, grad_s, b, grad_b = training_dataset.interpolate_superres(batch_output, resolution_factor=4)

		# Stack the corresponding outputs for easier processing
		batch_output = torch.cat([h, u, v, s, b], dim=1).cpu()

		# Store the batch output
		imfit_outputs[batch_slice] = batch_output

		# Process the next batch
		processed += this_batch

	# Compute the RMSE for each image
	rmse_per_image = torch.zeros(500, 5)

	for i in range(500):
		for j in range(5):
			rmse_per_image[i, j] = root_mean_square_error(sfere_outputs[i, j], imfit_outputs[i, j])

	rmse_per_image_mean_channel = torch.mean(rmse_per_image, dim=1)

	# Plot the resulting RMSE
	# plt.boxplot([
	# 	rmse_per_image_mean_channel[0:100],
	# 	rmse_per_image_mean_channel[100:200],
	# 	rmse_per_image_mean_channel[200:300],
	# 	rmse_per_image_mean_channel[300:400],
	# 	rmse_per_image_mean_channel[400:500],
	# ])

	plt.semilogy(rmse_per_image[:, 0], label="h")
	plt.semilogy(rmse_per_image[:, 1], label="u")
	plt.semilogy(rmse_per_image[:, 2], label="v")
	plt.semilogy(rmse_per_image[:, 3], label="s")
	plt.semilogy(rmse_per_image[:, 4], label="b")

	plt.legend()
	plt.show()



if __name__ == "__main__":
	main()