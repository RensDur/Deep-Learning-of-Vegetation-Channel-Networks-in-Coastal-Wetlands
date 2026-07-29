import torch
import matplotlib.pyplot as plt
import time
from typing import List


class VideoChannel:

    def __init__(self, name, value_range, cmap):

        self.name = name
        self.value_range = value_range
        self.cmap = cmap

    def min_val(self):
        return self.value_range[0]

    def max_val(self):
        return self.value_range[1]


class VideoPlayer:


    def __init__(self, width, height, channels: List[VideoChannel], num_frames, fps):

        self.width = width
        self.height = height
        self.channels = channels
        self.num_frames = num_frames
        self.fps = fps

        # Video buffers
        self.buffer = torch.zeros(self.num_frames, len(self.channels), self.height, self.width)
        self.buffer_counter = 0

        # Initial boundary overlay
        self.bound_overlay = torch.zeros(height, width, 4) # RGBA to support transparency

        #
        # Prepare Matplotlib Windows
        #
        plt.ion()

        self.figure = plt.figure(figsize=(20, 10))

        # Axis per channel
        self.axes = [plt.subplot2grid((1, len(self.channels)), (0, column), colspan=1) for column in range(len(self.channels))]

        # Optimise spacing
        plt.tight_layout()
        plt.subplots_adjust(
            left=0.05,
            right=1-0.05,
            top=1-0.05,
            bottom=0.05,
            hspace=0.3,
            wspace=0.025
        )

        # Disable axis numbers for image plots
        for i in range(len(self.channels)):
            self.axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Apply channel configurations
        for i in range(len(self.channels)):
            self.axes[i].set(title=self.channels[i].name)
            
        # Create image plots
        self.plots = [self.axes[i].imshow(self.buffer[0, i].detach().cpu().numpy(), cmap=self.channels[i].cmap, vmin=self.channels[i].min_val(), vmax=self.channels[i].max_val()) for i in range(len(self.channels))]

        # Create image overlays (for boundary visualisation)
        self.plots_overlays = [self.axes[i].imshow(self.bound_overlay) for i in range(len(self.channels))]

        # Color bars for each plot
        for i in range(len(self.channels)):
            plt.colorbar(self.plots[i])

        # Windowing state machine
        self.is_open = False
        self.playing = False
        self.current_frame = 0
        self.time_last_frame = time.time()
        self.time_per_frame = 1.0 / self.fps
        


    def set_training_loss(self, training_loss):
        pass

    def open(self):

        def __on_figure_close(event):
            self.is_open = False

        # In interactive mode, plt.show() immediately returns
        plt.show()

        # Connect the on-close event
        self.figure.canvas.mpl_connect('close_event', __on_figure_close)

        # Toggle the window open
        self.is_open = True

    def update(self):
        if self.is_open:

            # Prepare the next frame
            self.current_frame += 1
            if self.current_frame >= self.num_frames:
                self.current_frame = 0
            
            for i in range(len(self.channels)):
                self.plots[i].set_data(self.buffer[self.current_frame, i].detach().cpu().numpy())
                self.plots_overlays[i].set_data(self.bound_overlay)

            # Wait until the next frame should be presented
            while not time.time() >= self.time_last_frame + self.time_per_frame:
                pass
            
            # Plot the domain (update existing plot)
            # Draw updated values
            self.figure.canvas.draw()

            # UI Loop: process all pending UI events
            self.figure.canvas.flush_events()

    def close(self):
        # Toggling the window closed will stop the ui thread
        self.is_open = False

    def set_data(self, *args):
        # *args is there to stick to the contract that's set by the windows that are used in this project
        # The first len(channels) entries correspond to the image data per channel

        # This function simply stores the provided data in the next buffer entry
        for i in range(len(self.channels)):
            self.buffer[self.buffer_counter, i] = args[i]

        # The boundary overlay is assumed to never change over time and corresponds to the last entry in args
        self.bound_overlay[:, :, 3] = args[-1]

        self.buffer_counter += 1

    def append_loss(self, loss_h, loss_u, loss_v, loss_s, loss_b):
        # Unsupported and unneeded function for this class
        # Purely there for class contract
        pass