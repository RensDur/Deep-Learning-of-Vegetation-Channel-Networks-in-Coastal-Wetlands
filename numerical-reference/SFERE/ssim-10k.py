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
        self.axs[1, 2].legend()

        # Color bars
        plt.colorbar(self.water_plot)
        plt.colorbar(self.momentum_u_plot)
        plt.colorbar(self.momentum_v_plot)
        plt.colorbar(self.sediment_plot)
        plt.colorbar(self.vegetation_plot)

        # Slider functionality
        self.axs[2, 0].set_visible(False)
        self.axs[2, 1].set_visible(False)
        self.snapshot_slider = Slider(self.axs[2, 2], "Snapshot", 10_000, 1_000_000, valinit=500_000, valfmt="%d", valstep=10_000)
        self.snapshot_slider.on_changed(self.slider_update)
        self.current_selected_index = -1
        self.slider_update(500_000)
        
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

        if selected_index == self.current_selected_index:
            return

        self.set_data(
            self.snapshots[selected_index, 0],
            self.snapshots[selected_index, 1],
            self.snapshots[selected_index, 2],
            self.snapshots[selected_index, 3],
            self.snapshots[selected_index, 4],
        )

        self.current_selected_index = selected_index

    def update(self):
        if self.is_open:
            # Plot the domain (update existing plot)
            # Draw updated values
            # self.figure.canvas.draw()

            # UI Loop: process all pending UI events
            self.figure.canvas.flush_events()

            plt.pause(0.05)

    def close(self):
        # Toggling the window closed will stop the ui thread
        self.is_open = False

    def set_data(self, h, u, v, s, b):

        self.water_plot.set_data(h.cpu().numpy())
        self.momentum_u_plot.set_data(u.cpu().numpy())
        self.momentum_v_plot.set_data(v.cpu().numpy())
        self.sediment_plot.set_data(s.cpu().numpy())
        self.vegetation_plot.set_data(b.cpu().numpy())

        self.figure.canvas.draw_idle()

    def set_ssim_scores(self, ssim_xs, ssim_scores, label):
        self.axs[1, 2].plot(ssim_xs, ssim_scores, label=label)
        self.axs[1, 2].legend()



def main():

    # Find the GPU device for pytorch
    torch_device = torch.device('cpu')  # Default to CPU
    # Switch to MPS (Apple Metal) if available
    if torch.backends.mps.is_available():
        torch_device = torch.device('mps')
    # Or CUDA if we're on an Nvidia machine
    elif torch.cuda.is_available():
        torch_device = torch.device('cuda')
    print(f"Using torch device '{torch_device}'")

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

    # Compute SSIM of consequtive 10k intervals
    ssim_scores_h = [1]
    ssim_scores_u = [1]
    ssim_scores_v = [1]
    ssim_scores_s = [1]
    ssim_scores_b = [1]

    for i in range(1, dataset_size):
        prev = snapshots[(i-1):i].to(torch_device)
        curr = snapshots[i:(i+1)].to(torch_device)

        # SSIM
        ssim_scores_h.append(torch.mean(ssim(prev[:,0:1], curr[:,0:1], kernel_size=11)).cpu().item())
        ssim_scores_u.append(torch.mean(ssim(prev[:,1:2], curr[:,1:2], kernel_size=11)).cpu().item())
        ssim_scores_v.append(torch.mean(ssim(prev[:,2:3], curr[:,2:3], kernel_size=11)).cpu().item())
        ssim_scores_s.append(torch.mean(ssim(prev[:,3:4], curr[:,3:4], kernel_size=11)).cpu().item())
        ssim_scores_b.append(torch.mean(ssim(prev[:,4:5], curr[:,4:5], kernel_size=11)).cpu().item())

        # Standard MSE
        # def __normalize(prev, curr):
        #     prev = prev - torch.min(prev)
        #     prev = prev / torch.max(prev)

        #     curr = curr - torch.min(curr)
        #     curr = curr / torch.max(curr)
            
        #     return prev, curr

        # prev[:,0:1], curr[:,0:1] = __normalize(prev[:,0:1], curr[:,0:1])
        # prev[:,1:2], curr[:,1:2] = __normalize(prev[:,1:2], curr[:,1:2])
        # prev[:,2:3], curr[:,2:3] = __normalize(prev[:,2:3], curr[:,2:3])
        # prev[:,3:4], curr[:,3:4] = __normalize(prev[:,3:4], curr[:,3:4])
        # prev[:,4:5], curr[:,4:5] = __normalize(prev[:,4:5], curr[:,4:5])

        # ssim_scores_h.append(torch.mean(torch.pow(prev[:,0:1] - curr[:,0:1], 2)).cpu().item())
        # ssim_scores_u.append(torch.mean(torch.pow(prev[:,1:2] - curr[:,1:2], 2)).cpu().item())
        # ssim_scores_v.append(torch.mean(torch.pow(prev[:,2:3] - curr[:,2:3], 2)).cpu().item())
        # ssim_scores_s.append(torch.mean(torch.pow(prev[:,3:4] - curr[:,3:4], 2)).cpu().item())
        # ssim_scores_b.append(torch.mean(torch.pow(prev[:,4:5] - curr[:,4:5], 2)).cpu().item())

    # Open the window
    win = Window(800, 800, snapshots)
    win.set_ssim_scores(selected_outputs, ssim_scores_h, "h")
    win.set_ssim_scores(selected_outputs, ssim_scores_u, "u")
    win.set_ssim_scores(selected_outputs, ssim_scores_v, "v")
    win.set_ssim_scores(selected_outputs, ssim_scores_s, "s")
    win.set_ssim_scores(selected_outputs, ssim_scores_b, "b")

    win.open()

    while win.is_open:
        win.update()



if __name__ == "__main__":
    main()