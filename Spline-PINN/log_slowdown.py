import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle


def count_within_range(x, min, max):
	x = x[torch.where(x <= max)]
	return x[torch.where(x >= min)].shape[0]

def find_closest_x(y, xs, ys):
	closest_index = 0
	for i in range(len(ys)):
		if ys[i] >= y:
			closest_index = i
			break
	
	return xs[closest_index]

def compute_curve(steepness, max_iter, total_samples):
	xs = torch.linspace(0, steepness, total_samples)
	ys = torch.exp(xs)
	ys = ys - torch.min(ys)
	ys = ys / torch.max(ys) * max_iter
	xs = xs / steepness * total_samples

	return xs, ys


def main():

	# Create subplots
	fig, ax = plt.subplots()
	# plt.subplots_adjust(bottom=1/4)

	# steepness = [0.1, 1, 3, 5, 7]
	# colors = ["#00559933", "#00559966", "#00559999", "#005599cc", "#005599ff"]

	steepness = [3.3]
	colors = ["#005599ff"]

	max_iter = 1_000_000

	samples_per_pinn = 100
	total_samples = samples_per_pinn * 5

	for i, s in enumerate(steepness):
		xs, ys = compute_curve(s, max_iter, total_samples)

		curve, = plt.plot(xs, ys, color=colors[i])

	stages = [
		0,
		30_000,
		100_000,
		250_000,
		500_000,
		1_000_000
	]

	# plt.hlines(stages, 0, total_samples, color="red")
	# plt.vlines([100, 200, 300, 400], 0, max_iter, color="red", linestyles="dashed")

	plt.hlines([ys[100-1]], 0, 100, color="red")
	plt.hlines([ys[200-1]], 0, 200, color="red")
	plt.hlines([ys[300-1]], 0, 300, color="red")
	plt.hlines([ys[400-1]], 0, 400, color="red")
	plt.hlines([ys[500-1]], 0, 500, color="red")

	print([ys[100-1]])
	print([ys[200-1]])
	print([ys[300-1]])
	print([ys[400-1]])
	print([ys[500-1]])

	plt.vlines([100], 0, ys[100-1], color="red", linestyles="dashed")
	plt.vlines([200], 0, ys[200-1], color="red", linestyles="dashed")
	plt.vlines([300], 0, ys[300-1], color="red", linestyles="dashed")
	plt.vlines([400], 0, ys[400-1], color="red", linestyles="dashed")
	plt.vlines([500], 0, ys[500-1], color="red", linestyles="dashed")

	plt.title("Simulation time per sample")
	plt.xlabel("Sample index n")
	plt.ylabel("Simulation time k(n) (Number of iterations)")

	xticks = [0, 100, 200, 300, 400, 500]
	xlabels = ["0", "N/5", "2N/5", "3N/5", "4N/5", "N"]

	plt.xticks(xticks, labels=xlabels)

	yticks = stages
	ylabels = ["0", "30K", "100K", "250K", "500K", "1M"]

	plt.yticks(yticks, labels=ylabels)
	
	if False:
		axis_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
		slider_steepness = Slider(axis_slider, "Steepness", 0.1, 10, valinit=steepness, valstep=0.01)

		def update(val):
			steepness = slider_steepness.val

			xs, ys = compute_curve(steepness, max_iter, total_samples)

			# vlines.set_data(samples_per_stage)
			curve.set_ydata(ys)
			fig.canvas.draw_idle()

		slider_steepness.on_changed(update)

	plt.show()




if __name__ == "__main__":
	main()





# import matplotlib.pyplot as plt
# import torch

# steepness = 3.5
# max_iter = 1_000_000
# total_samples = 1000

# xs = torch.linspace(0, steepness, total_samples)
# ys = torch.exp(xs)
# ys = ys - torch.min(ys)
# ys = ys / torch.max(ys) * max_iter
# xs = xs / steepness * max_iter


# count_stage1 = (ys[torch.where(ys < 30_000)]).shape[0]
# count_stage2 = (ys[torch.where(ys < 100_000)]).shape[0] - count_stage1
# count_stage3 = (ys[torch.where(ys < 250_000)]).shape[0] - count_stage2 - count_stage1
# count_stage4 = (ys[torch.where(ys < 500_000)]).shape[0] - count_stage3 - count_stage2 - count_stage1
# count_stage5 = (ys).shape[0] - count_stage4 - count_stage3 - count_stage2 - count_stage1

# print(f"Number of samples below 30k: {count_stage1}")
# print(f"Number of samples below 100k: {count_stage2}")
# print(f"Number of samples below 250k: {count_stage3}")
# print(f"Number of samples below 500k: {count_stage4}")
# print(f"Number of samples below 1m: {count_stage5}")



# # plt.hist(ys, bins=[0, 30_000, 100_000, 250_000, 500_000, 1_000_000])
# plt.vlines([0, 30_000, 100_000, 250_000, 500_000, 1_000_000], 0, max_iter, colors="red")
# plt.hlines([0, 30_000, 100_000, 250_000, 500_000, 1_000_000], 0, max_iter, colors="red")
# plt.plot(xs, ys)
# plt.show()


