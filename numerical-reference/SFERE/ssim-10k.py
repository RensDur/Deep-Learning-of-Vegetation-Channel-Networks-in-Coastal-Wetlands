import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ssim import *



class Window:

    def __init__(self, width, height, snapshots):

        self.width = width
        self.height = height

        # Dataset
        self.snapshots = snapshots # Torch tensor [NUM_IMGS, 5{h,u,v,s,b}, height, width]
        self.num_snapshots = self.snapshots.shape[0]

        # Initial blank images
        self.h = torch.zeros(1, 1, height, width)
        self.u = torch.zeros(1, 1, height, width)
        self.v = torch.zeros(1, 1, height, width)
        self.s = torch.zeros(1, 1, height, width)
        self.b = torch.zeros(1, 1, height, width)

        # Matplotlib interactive mode
        plt.ion()

        # Create subplots
        self.figure, self.axs = plt.subplots(3, 3, figsize=(20, 10), height_ratios=[1, 1, 0.1])

        self.water_plot = self.axs[0, 0].imshow(self.h[0,0].clone().detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=0.2)
        self.momentum_u_plot = self.axs[0, 1].imshow(self.u[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.5, vmax=0.5)
        self.momentum_v_plot = self.axs[0, 2].imshow(self.v[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.5, vmax=0.5)
        self.sediment_plot = self.axs[1, 0].imshow(self.s[0,0].clone().detach().cpu().numpy(), cmap="YlOrBr", vmin=0, vmax=0.2)
        self.vegetation_plot = self.axs[1, 1].imshow(self.b[0,0].clone().detach().cpu().numpy(), cmap="YlGn", vmin=0, vmax=1500)

        # Title and axes configuration
        self.axs[0, 0].set(title="Water Layer Thickness", xlabel="Cross shore", ylabel="Along shore")
        self.axs[0, 1].set(title="Momentum u (x-direction)", xlabel="Cross shore", ylabel="Along shore")
        self.axs[0, 2].set(title="Momentum v (y-direction)", xlabel="Cross shore", ylabel="Along shore")
        self.axs[1, 0].set(title="Sediment bed", xlabel="Cross shore", ylabel="Along shore")
        self.axs[1, 1].set(title="Vegetation density", xlabel="Cross shore", ylabel="Along shore")
        self.axs[1, 2].set(title="SSIM Similarity between snapshots", xlabel="Iter. count", ylabel="SSIM")

        # Color bars
        plt.colorbar(self.water_plot)
        plt.colorbar(self.momentum_u_plot)
        plt.colorbar(self.momentum_v_plot)
        plt.colorbar(self.sediment_plot)
        plt.colorbar(self.vegetation_plot)

        # Slider functionality
        self.axs[2, 0].set_visible(False)
        self.axs[2, 1].set_visible(False)
        self.snapshot_slider = Slider(self.axs[2, 2], "Snapshot", 10_000, 1_000_000, valfmt="%d", valstep=10_000)
        self.snapshot_slider.on_changed(self.slider_update)
        
        # Window status
        self.is_open = False


    def open(self):

        def __on_figure_close(event):
            self.is_open = False

        # In interactive mode, plt.show() immediately returns
        plt.show()

        # Connect the on-close event
        self.figure.canvas.mpl_connect('close_event', __on_figure_close)

        # Toggle the window open
        self.is_open = True

    def slider_update(self, slider_val):
        selected_index = int(slider_val / 10_000) - 1
        self.set_data(
            self.snapshots[selected_index, 0],
            self.snapshots[selected_index, 1],
            self.snapshots[selected_index, 2],
            self.snapshots[selected_index, 3],
            self.snapshots[selected_index, 4],
        )

    def update(self):
        if self.is_open:
            # Plot the domain (update existing plot)
            # Draw updated values
            self.figure.canvas.draw()

            # UI Loop: process all pending UI events
            self.figure.canvas.flush_events()

    def close(self):
        # Toggling the window closed will stop the ui thread
        self.is_open = False

    def set_data(self, h, u, v, s, b):

        self.water_plot.set_data(h.detach().cpu().numpy())
        self.momentum_u_plot.set_data(u.detach().cpu().numpy())
        self.momentum_v_plot.set_data(v.detach().cpu().numpy())
        self.sediment_plot.set_data(s.detach().cpu().numpy())
        self.vegetation_plot.set_data(b.detach().cpu().numpy())



def main():

    # Select which images to load
    start_index = 10_000
    end_index = 1_000_000
    selected_outputs = [i for i in range(start_index, end_index+1, 10_000)]
    dataset_size = len(selected_outputs)
    
    # Load all SFERE snapshots with 10k iteration intervals from disk
    snapshots = torch.zeros(
        dataset_size,
        5,
        800,
        800
    )

    for i, snapshot in enumerate(selected_outputs):
        h = torch.load(f"./snapshots-10k/{snapshot}/h.pt").cpu()
        u = torch.load(f"./snapshots-10k/{snapshot}/u.pt").cpu()
        v = torch.load(f"./snapshots-10k/{snapshot}/v.pt").cpu()
        s = torch.load(f"./snapshots-10k/{snapshot}/s.pt").cpu()
        b = torch.load(f"./snapshots-10k/{snapshot}/b.pt").cpu()

        compound = torch.cat([h, u, v, s, b], dim=1)

        snapshots[i] = compound

    # Open the window
    win = Window(800, 800, snapshots)

    win.open()

    while win.is_open:
        win.update()



if __name__ == "__main__":
    main()