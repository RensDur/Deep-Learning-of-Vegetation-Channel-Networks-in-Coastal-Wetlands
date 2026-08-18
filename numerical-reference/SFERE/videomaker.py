import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



def main():
    global is_open

    # Load all the snapshots from disk
    snapshots = torch.zeros(500, 5, 800, 800)


    for i in range(500):
        snapshots[i, 0] = torch.load(f"./snapshots-log-slowdown/snapshot_{i}/h.pt").detach().cpu()
        snapshots[i, 1] = torch.load(f"./snapshots-log-slowdown/snapshot_{i}/u.pt").detach().cpu()
        snapshots[i, 2] = torch.load(f"./snapshots-log-slowdown/snapshot_{i}/v.pt").detach().cpu()
        snapshots[i, 3] = torch.load(f"./snapshots-log-slowdown/snapshot_{i}/s.pt").detach().cpu()
        snapshots[i, 4] = torch.load(f"./snapshots-log-slowdown/snapshot_{i}/b.pt").detach().cpu()

    
    h_threshold = 0.0005
    b_threshold = 100
    h_vmax = 0.1
    b_vmax = 1500
    
    h_norm = Normalize(vmin=h_threshold, vmax=h_vmax, clip=True)
    b_norm = Normalize(vmin=b_threshold, vmax=b_vmax, clip=True)
    
    def h_alpha(h_image):
        # smooth fade instead of a hard mask cutoff
        a = h_norm(h_image)
        return np.clip(a, 0, 1) * 0.6  # cap max opacity at 0.6 instead of flat 0.3
    
    def b_alpha(b_image):
        a = b_norm(b_image)
        return np.clip(a, 0, 1)
    
    plt.ion()
    figure, axes = plt.subplots()
    
    s0 = snapshots[0, 3].numpy()
    h0 = snapshots[0, 0].numpy()
    b0 = snapshots[0, 4].numpy()
    
    s_plot = axes.imshow(s0, cmap="YlOrBr", vmin=0, vmax=0.5, animated=True)
    h_plot = axes.imshow(h0, cmap="Blues", vmin=0, vmax=h_vmax,
                          alpha=h_alpha(h0), animated=True)
    b_plot = axes.imshow(b0, cmap="YlGn", vmin=0, vmax=b_vmax,
                          alpha=b_alpha(b0), animated=True)
    
    figure.canvas.draw()
    background = figure.canvas.copy_from_bbox(axes.bbox)

    def __on_figure_close(event):
        global is_open
        is_open = False

    # Connect the on-close event
    figure.canvas.mpl_connect('close_event', __on_figure_close)

    # Toggle the window open
    is_open = True
    
    for current_snapshot in range(500):
        s_image = snapshots[current_snapshot, 3].numpy()
        h_image = snapshots[current_snapshot, 0].numpy()
        b_image = snapshots[current_snapshot, 4].numpy()
    
        s_plot.set_data(s_image)
        h_plot.set_data(h_image)
        h_plot.set_alpha(h_alpha(h_image))
        b_plot.set_data(b_image)
        b_plot.set_alpha(b_alpha(b_image))
    
        figure.canvas.restore_region(background)
        axes.draw_artist(s_plot)
        axes.draw_artist(h_plot)
        axes.draw_artist(b_plot)
        figure.canvas.blit(axes.bbox)
        figure.canvas.flush_events()


    while is_open:
        figure.canvas.draw_idle()
        figure.canvas.flush_events()

        




if __name__ == "__main__":
    main()