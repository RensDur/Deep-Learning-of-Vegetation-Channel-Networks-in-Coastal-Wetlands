import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from imfit_general import CompoundFitNet, FitDataset
from spline.spline_variable import SplineVariable
from spline.spline_array import SplineArray
import lpips

# Perceptual metric LPIPS loss function
# LPIPS by
loss_fn_alex = lpips.LPIPS(net='alex')

def root_relative_square_error(img1, img2):
    """
    Compute the root relative mean square error between img1 and img2
    img1 is the reference image that determines the pixel range
    """
    img1_mean = torch.mean(img1)

    numerator = torch.sum(torch.pow(img1 - img2, 2))
    denominator = torch.sum(torch.pow(img1 - img1_mean, 2))

    return torch.pow(numerator / denominator, 0.5)


def compute_RRSE(sfere_outputs, training_dataset, imfit_net, torch_device):

    # Process the images in batches of 50
	batch_size = 4
	processed = 0
	num_images = sfere_outputs.shape[0]
	imfit_outputs = torch.zeros_like(sfere_outputs)

	while processed < num_images:
		this_batch = min(batch_size, num_images - processed)
		batch_slice = slice(processed, processed + this_batch)
		print(f"\rProcessing batch {batch_slice}", end="")

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

	print("")

	# Compute the Root Relative Square Error for each image (separately stacked as 5 channels: h, u, v, S, B)
	rrse_per_image = torch.zeros(num_images, 5)

	for i in range(num_images):
		for j in range(5):
			rrse_per_image[i, j] = root_relative_square_error(sfere_outputs[i, j], imfit_outputs[i, j])

	# Compute the combined RRSE per image (mean of the 5 channels per image)
	rrse_per_sample = torch.mean(rrse_per_image, dim=1)

	return rrse_per_sample

def compute_LPIPS(sfere_outputs, training_dataset, imfit_net, torch_device):

    # Process the images in batches of 50
	batch_size = 4
	processed = 0
	num_images = sfere_outputs.shape[0]
	imfit_outputs = torch.zeros_like(sfere_outputs)

	while processed < num_images:
		this_batch = min(batch_size, num_images - processed)
		batch_slice = slice(processed, processed + this_batch)
		print(f"\rProcessing batch {batch_slice}", end="")

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

	print("")

	# Compute the Root Relative Square Error for each image (separately stacked as 5 channels: h, u, v, S, B)
	lpips_per_image = torch.zeros(num_images, 5)

	for i in range(num_images):
		for j in range(5):
			lpips_per_image[i, j] = loss_fn_alex(sfere_outputs[i, j], imfit_outputs[i, j])

	# Compute the combined LPIPS per image (mean of the 5 channels per image)
	lpips_per_sample = torch.mean(lpips_per_image, dim=1)

	return lpips_per_sample


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

    # Load all sfere samples (validation set) from disk
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

    # rrse_per_training_sample = compute_RRSE(training_dataset.numerical_output_states[0:100], training_dataset, imfit_net, torch_device)
    rrse_per_validation_sample = compute_RRSE(sfere_outputs, training_dataset, imfit_net, torch_device)

    print(f"Mean training error: {torch.mean(rrse_per_validation_sample[1:])}")

    # Plot the resulting Root Relative Square Error per sample
    fig, ax = plt.subplots()

    plt.title("Relative Image Fitting Error per Sample")
    plt.xlabel("Sample index n")
    plt.ylabel("Root Relative Square Error (RRSE) [log]")

    ax.vlines([0, 100, 200, 300, 400, 500], ymin=0, ymax=torch.max(rrse_per_validation_sample[1:]), colors="#cccccc55", linestyles="dashed")

    ax.plot(rrse_per_validation_sample, label="Relative Error")

    m1 = torch.mean(rrse_per_validation_sample[1:100])
    m2 = torch.mean(rrse_per_validation_sample[100:200])
    m3 = torch.mean(rrse_per_validation_sample[200:300])
    m4 = torch.mean(rrse_per_validation_sample[300:400])
    m5 = torch.mean(rrse_per_validation_sample[400:500])

    ax.scatter(
    	[50, 150, 250, 350, 450],
    	[m1, m2, m3, m4, m5],
    	s=100, facecolors="none", edgecolors="C1", linewidths=1.5, label="Mean per category"
    )

    ax.annotate(f"{m1:.3f}", xy=(50, m1), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{m2:.3f}", xy=(150, m2), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{m3:.3f}", xy=(250, m3), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{m4:.3f}", xy=(350, m4), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{m5:.3f}", xy=(450, m5), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)

    print(f"Average of m1 m2 m3 m4 m5: {torch.mean(torch.Tensor([m1, m2, m3, m4, m5]))}")

    ax.set_yscale('log')

    plt.legend(facecolor="#ffffff88")
    plt.show()



if __name__ == "__main__":
	main()
