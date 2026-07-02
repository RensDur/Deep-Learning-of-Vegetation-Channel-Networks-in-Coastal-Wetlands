import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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
	xs = xs / steepness * max_iter

	return xs, ys


def main():

	steepness = 3.5
	max_iter = 1_000_000

	samples_per_pinn = 100
	total_samples = samples_per_pinn * 5

	xs, ys = compute_curve(steepness, max_iter, total_samples)

	stages = [
		0,
		30_000,
		100_000,
		250_000,
		500_000,
		1_000_000
	]

	# Compute sample counts per bucket
	samples_per_stage = [find_closest_x(s, xs, ys) for s in stages]

	# Create subplots
	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=1/4)

	vlines = plt.vlines(samples_per_stage, 0, max_iter, color="red")

	curve, = plt.plot(xs, ys)
	
	axis_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
	slider_steepness = Slider(axis_slider, "Steepness", 0.1, 10, valinit=steepness, valstep=0.01)

	def update(val):
		steepness = slider_steepness.val

		xs, ys = compute_curve(steepness, max_iter, total_samples)

		samples_per_stage = [find_closest_x(s, xs, ys) for s in stages]

		vlines.set_data(samples_per_stage)
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


