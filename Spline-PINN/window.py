import numpy as np
import time
import colormaps

import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec




class PerformanceSummaryWindow:

    def __init__(self, width, height, stages, interval):

        self.width = width
        self.height = height
        self.stages = stages
        self.interval = interval

        self.current_stage = 0

        # Initial blank images
        self.h = torch.zeros(self.stages, 1, height, width)
        self.u = torch.zeros(self.stages, 1, height, width)
        self.v = torch.zeros(self.stages, 1, height, width)
        self.s = torch.zeros(self.stages, 1, height, width)
        self.b = torch.zeros(self.stages, 1, height, width)

        # Keep track of evaluation loss
        self.h_loss_data = np.array([])
        self.u_loss_data = np.array([])
        self.v_loss_data = np.array([])
        self.s_loss_data = np.array([])
        self.b_loss_data = np.array([])

        # Matplotlib interactive mode
        plt.ion()

        # Create window for training loss
        self.loss_figure = plt.figure(figsize=(5, 5))

        # Create subplots
        self.figure = plt.figure(figsize=(20, 10))

        self.h_axs = [plt.subplot2grid((6, self.stages), (0, col), colspan=1) for col in range(self.stages)]
        self.u_axs = [plt.subplot2grid((6, self.stages), (1, col), colspan=1) for col in range(self.stages)]
        self.v_axs = [plt.subplot2grid((6, self.stages), (2, col), colspan=1) for col in range(self.stages)]
        self.s_axs = [plt.subplot2grid((6, self.stages), (3, col), colspan=1) for col in range(self.stages)]
        self.b_axs = [plt.subplot2grid((6, self.stages), (4, col), colspan=1) for col in range(self.stages)]
        self.loss_ax = plt.subplot2grid((6, self.stages), (5, 0), colspan=self.stages)
        plt.tight_layout()

        # Custom spacing
        plt.subplots_adjust(
            left=0.05,
            right=1-0.05,
            top=1-0.05,
            bottom=0.05,
            hspace=0.3,
            wspace=0.025
        )

        # Disable axis numbers for image plots
        for i in range(self.stages):
            self.h_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.u_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.v_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.s_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.b_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Titles and labels
        self.h_axs[0].set(title=f"Iteration: 0", ylabel="h")
        for i in range(1, self.stages):
            self.h_axs[i].set(title=f"{i*self.interval}")

        self.u_axs[0].set(ylabel="u")
        self.v_axs[0].set(ylabel="v")
        self.s_axs[0].set(ylabel="s")
        self.b_axs[0].set(ylabel="b")

        # Set loss axis limits
        self.loss_ax.set_xlim([-self.interval/2, self.stages * self.interval - self.interval/2])
        self.loss_ax.set_ylim([-20, 20])

        # Set loss axis tick positions
        self.loss_ax.set_xticks(np.arange(0, self.stages * self.interval, self.interval))

        # Create all the image plots
        self.h_img_plots = [self.h_axs[col].imshow(self.h[col,0].detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=0.5) for col in range(self.stages)]
        self.u_img_plots = [self.u_axs[col].imshow(self.u[col,0].detach().cpu().numpy(), cmap="bwr", vmin=-1, vmax=1) for col in range(self.stages)]
        self.v_img_plots = [self.v_axs[col].imshow(self.v[col,0].detach().cpu().numpy(), cmap="bwr", vmin=-1, vmax=1) for col in range(self.stages)]
        self.s_img_plots = [self.s_axs[col].imshow(self.s[col,0].detach().cpu().numpy(), cmap="YlOrBr", vmin=0, vmax=0.5) for col in range(self.stages)]
        self.b_img_plots = [self.b_axs[col].imshow(self.b[col,0].detach().cpu().numpy(), cmap="YlGn", vmin=0,  vmax=1500) for col in range(self.stages)]

        # Create loss plots
        self.h_loss_plot = self.loss_ax.plot(range(self.h_loss_data.shape[0]), self.h_loss_data, label="h-loss")[0]
        self.u_loss_plot = self.loss_ax.plot(range(self.u_loss_data.shape[0]), self.u_loss_data, label="u-loss")[0]
        self.v_loss_plot = self.loss_ax.plot(range(self.v_loss_data.shape[0]), self.v_loss_data, label="v-loss")[0]
        self.s_loss_plot = self.loss_ax.plot(range(self.s_loss_data.shape[0]), self.s_loss_data, label="s-loss")[0]
        self.b_loss_plot = self.loss_ax.plot(range(self.b_loss_data.shape[0]), self.b_loss_data, label="b-loss")[0]
        self.loss_ax.legend()
        self.loss_ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
        
        # Color bars
        plt.colorbar(self.h_img_plots[-1])
        plt.colorbar(self.u_img_plots[-1])
        plt.colorbar(self.v_img_plots[-1])
        plt.colorbar(self.s_img_plots[-1])
        plt.colorbar(self.b_img_plots[-1])

        # Window status
        self.is_open = False

    def set_training_loss(self, training_loss):
        plt_ax = self.loss_figure.add_subplot(111)
        training_loss.plot(ax=plt_ax)
        plt_ax.set(
            title="Training Loss",
            xlabel="Training iteration",
            ylabel="Log square loss (bias +1E-4)"
        )

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

            # Plot training loss
            self.loss_figure.canvas.draw()
            self.loss_figure.canvas.flush_events()

            # Plot the domain (update existing plot)
            # Draw updated values
            self.figure.canvas.draw()

            # UI Loop: process all pending UI events
            self.figure.canvas.flush_events()

    def close(self):
        # Toggling the window closed will stop the ui thread
        self.is_open = False

    def set_data(self, h, u, v, s, b, index):

        if index % self.interval == 0:
            stage = self.current_stage

            if stage == self.stages:
                self.current_stage = 0
                stage = 0

            if stage < self.stages:
                self.current_stage += 1
                self.h_img_plots[stage].set_data(h.detach().cpu().numpy())
                self.u_img_plots[stage].set_data(u.detach().cpu().numpy())
                self.v_img_plots[stage].set_data(v.detach().cpu().numpy())
                self.s_img_plots[stage].set_data(s.detach().cpu().numpy())
                self.b_img_plots[stage].set_data(b.detach().cpu().numpy())

    def append_loss(self, loss_h, loss_u, loss_v, loss_s, loss_b):

        # Append to data
        self.h_loss_data = np.append(self.h_loss_data, np.array([loss_h.detach().cpu().numpy()]))
        self.u_loss_data = np.append(self.u_loss_data, np.array([loss_u.detach().cpu().numpy()]))
        self.v_loss_data = np.append(self.v_loss_data, np.array([loss_v.detach().cpu().numpy()]))
        self.s_loss_data = np.append(self.s_loss_data, np.array([loss_s.detach().cpu().numpy()]))
        self.b_loss_data = np.append(self.b_loss_data, np.array([loss_b.detach().cpu().numpy()]))

        # Set X Data
        self.h_loss_plot.set_xdata(range(self.h_loss_data.shape[0]))
        self.u_loss_plot.set_xdata(range(self.u_loss_data.shape[0]))
        self.v_loss_plot.set_xdata(range(self.v_loss_data.shape[0]))
        self.s_loss_plot.set_xdata(range(self.s_loss_data.shape[0]))
        self.b_loss_plot.set_xdata(range(self.b_loss_data.shape[0]))

        # Set Y Data
        self.h_loss_plot.set_ydata(self.h_loss_data)
        self.u_loss_plot.set_ydata(self.u_loss_data)
        self.v_loss_plot.set_ydata(self.v_loss_data)
        self.s_loss_plot.set_ydata(self.s_loss_data)
        self.b_loss_plot.set_ydata(self.b_loss_data)

        # Update the axis limits
        min_y = np.min(np.array([
            np.min(self.h_loss_data),
            np.min(self.u_loss_data),
            np.min(self.v_loss_data),
            np.min(self.s_loss_data),
            np.min(self.b_loss_data),
        ])).item()

        max_y = np.max(np.array([
            np.max(self.h_loss_data),
            np.max(self.u_loss_data),
            np.max(self.v_loss_data),
            np.max(self.s_loss_data),
            np.max(self.b_loss_data),
        ])).item()

        min_max_range = max_y - min_y
        min_y = min_y - 0.1*min_max_range
        max_y = max_y + 0.1*min_max_range

        self.loss_ax.set_ylim([min_y, max_y])

class PerformanceSummaryWindow_Hydrology:

    def __init__(self, width, height, stages, interval, print_loss_images=False, params=None):

        self.width = width
        self.height = height
        self.stages = stages
        self.interval = interval

        self.current_stage = 0

        self.params = params

        # Initial blank images
        self.h = torch.zeros(self.stages, 1, height, width)
        self.u = torch.zeros(self.stages, 1, height, width)
        self.v = torch.zeros(self.stages, 1, height, width)
        self.s = torch.zeros(self.stages, 1, height, width)
        self.b = torch.zeros(self.stages, 1, height, width)

        # Keep track of evaluation loss
        self.h_loss_data = np.array([])
        self.u_loss_data = np.array([])
        self.v_loss_data = np.array([])
        self.s_loss_data = np.array([])
        self.b_loss_data = np.array([])

        # Matplotlib interactive mode
        plt.ion()
        
        fig_width = 12.0
        fig_height = 6.0
        left = 0.9
        right = 0.6
        top = 0.4
        bottom = 0.4

        # Create window for training loss
        self.loss_figure = plt.figure(figsize=(5, 5))

        # Create subplots
        self.figure = plt.figure(figsize=(fig_width, fig_height))

        width_ratios = [1, 0.4] + [1] * (self.stages-1) + [0.1] # [0.1] is for the color bars. The first and last column each get a color bar
        self.grid_spec = GridSpec(3, self.stages+2, width_ratios=width_ratios, figure=self.figure) # Add one for the color bars

        self.h_axs = [self.figure.add_subplot(self.grid_spec[0, col]) for col in range(self.stages+1) if not col == 1] # Skip the color bar column
        self.u_axs = [self.figure.add_subplot(self.grid_spec[1, col]) for col in range(self.stages+1) if not col == 1]
        self.v_axs = [self.figure.add_subplot(self.grid_spec[2, col]) for col in range(self.stages+1) if not col == 1]
        # self.s_axs = [plt.subplot2grid((6, self.stages), (3, col), colspan=1) for col in range(self.stages)]
        # self.b_axs = [plt.subplot2grid((6, self.stages), (4, col), colspan=1) for col in range(self.stages)]
        # self.loss_ax = plt.subplot2grid((6, self.stages), (5, 0), colspan=self.stages)

        # Separate column for color bars
        self.h_cax_1_fullspan = self.figure.add_subplot(self.grid_spec[0, 1])
        self.u_cax_1_fullspan = self.figure.add_subplot(self.grid_spec[1, 1])
        self.v_cax_1_fullspan = self.figure.add_subplot(self.grid_spec[2, 1])

        self.h_cax_1_fullspan.axis("off")
        self.u_cax_1_fullspan.axis("off")
        self.v_cax_1_fullspan.axis("off")

        self.h_cax_1 = self.h_cax_1_fullspan.inset_axes([0, 0, 0.25, 1])
        self.u_cax_1 = self.u_cax_1_fullspan.inset_axes([0, 0, 0.25, 1])
        self.v_cax_1 = self.v_cax_1_fullspan.inset_axes([0, 0, 0.25, 1])
        
        self.h_cax_2 = self.figure.add_subplot(self.grid_spec[0, self.stages+1])
        self.u_cax_2 = self.figure.add_subplot(self.grid_spec[1, self.stages+1])
        self.v_cax_2 = self.figure.add_subplot(self.grid_spec[2, self.stages+1])

        # Custom spacing
        left   = left   / fig_width
        right  = 1 - right / fig_width
        bottom = bottom / fig_height
        top    = 1 - top / fig_height
        plt.subplots_adjust(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            hspace=0.15,
            wspace=0.01
        )

        # Disable axis numbers for image plots
        for i in range(self.stages):
            self.h_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.u_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self.v_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # self.s_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # self.b_axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Titles and labels
        self.h_axs[0].set(title=f"Starting condition")
        for i in range(1, self.stages):
            self.h_axs[i].set(title=f"{i*self.interval}")

        self.h_axs[1].set(title=f"Iteration: {1*self.interval}")

        self.h_axs[0].set_ylabel("Water level ($h$)")
        self.u_axs[0].set_ylabel("Flow velocity [x] ($u$)")
        self.v_axs[0].set_ylabel("Flow velocity [y] ($v$)")
        # self.s_axs[0].set(ylabel="s")
        # self.b_axs[0].set(ylabel="b")

        # Set loss axis limits
        # self.loss_ax.set_xlim([-self.interval/2, self.stages * self.interval - self.interval/2])
        # self.loss_ax.set_ylim([-20, 20])

        # Set loss axis tick positions
        # self.loss_ax.set_xticks(np.arange(0, self.stages * self.interval, self.interval))

        # Create all the image plots
        if print_loss_images:
            self.h_img_plots = [self.h_axs[col].imshow(self.h[col,0].detach().cpu().numpy(), cmap="gray", vmin=-10) for col in range(self.stages)]
            self.u_img_plots = [self.u_axs[col].imshow(self.u[col,0].detach().cpu().numpy(), cmap="gray", vmin=-10) for col in range(self.stages)]
            self.v_img_plots = [self.v_axs[col].imshow(self.v[col,0].detach().cpu().numpy(), cmap="gray", vmin=-10) for col in range(self.stages)]
        else:
            
            self.h_img_plots = [self.h_axs[col].imshow(self.h[col,0].detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=0.1) for col in range(self.stages)]
            self.u_img_plots = [self.u_axs[col].imshow(self.u[col,0].detach().cpu().numpy(), cmap="bwr", vmin=-1, vmax=1) for col in range(self.stages)]
            self.v_img_plots = [self.v_axs[col].imshow(self.v[col,0].detach().cpu().numpy(), cmap="bwr", vmin=-1, vmax=1) for col in range(self.stages)]

            # Adjust the color limits of the first plots, as they are generally narrower in SFERE compared to our PINNs
            self.h_img_plots[0].set_clim(vmin=0, vmax=0.05)
            self.u_img_plots[0].set_clim(vmin=-0.2, vmax=0.2)
            self.v_img_plots[0].set_clim(vmin=-0.2, vmax=0.2)
            
        # self.s_img_plots = [self.s_axs[col].imshow(self.s[col,0].detach().cpu().numpy(), cmap="YlOrBr", vmin=0, vmax=0.2) for col in range(self.stages)]
        # self.b_img_plots = [self.b_axs[col].imshow(self.b[col,0].detach().cpu().numpy(), cmap="YlGn", vmin=0,  vmax=1400) for col in range(self.stages)]

        # Create loss plots
        # self.h_loss_plot = self.loss_ax.plot(range(self.h_loss_data.shape[0]), self.h_loss_data, label="h-loss")[0]
        # self.u_loss_plot = self.loss_ax.plot(range(self.u_loss_data.shape[0]), self.u_loss_data, label="u-loss")[0]
        # self.v_loss_plot = self.loss_ax.plot(range(self.v_loss_data.shape[0]), self.v_loss_data, label="v-loss")[0]
        # self.s_loss_plot = self.loss_ax.plot(range(self.s_loss_data.shape[0]), self.s_loss_data, label="s-loss")[0]
        # self.b_loss_plot = self.loss_ax.plot(range(self.b_loss_data.shape[0]), self.b_loss_data, label="b-loss")[0]
        # self.loss_ax.legend()
        # self.loss_ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
        
        # Color bars
        plt.colorbar(self.h_img_plots[0], cax=self.h_cax_1)
        plt.colorbar(self.u_img_plots[0], cax=self.u_cax_1)
        plt.colorbar(self.v_img_plots[0], cax=self.v_cax_1)
        plt.colorbar(self.h_img_plots[-1], cax=self.h_cax_2)
        plt.colorbar(self.u_img_plots[-1], cax=self.u_cax_2)
        plt.colorbar(self.v_img_plots[-1], cax=self.v_cax_2)
        # plt.colorbar(self.s_img_plots[-1])
        # plt.colorbar(self.b_img_plots[-1])

        # Window status
        self.is_open = False

    def set_training_loss(self, training_loss):
        training_loss.plot(ax=self.loss_figure.add_subplot(111))

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

            # Plot training loss
            self.loss_figure.canvas.draw()
            self.loss_figure.canvas.flush_events()

            # Plot the domain (update existing plot)
            # Draw updated values
            self.figure.canvas.draw()

            # UI Loop: process all pending UI events
            self.figure.canvas.flush_events()

    def close(self):
        # Toggling the window closed will stop the ui thread
        self.is_open = False

    def set_data(self, h, u, v, s, b, index):

        if index % self.interval == 0:
            stage = self.current_stage

            if stage == self.stages:
                self.current_stage = 0
                stage = 0

                self.figure.savefig(f"./Hybrid Hydro-PINN evaluation/figures/Hybrid Hydro-PINN solution {self.params.sfere_start}-{self.params.sfere_end}.jpg", dpi=150)

            if stage < self.stages:
                self.current_stage += 1
                self.h_img_plots[stage].set_data(h.detach().cpu().numpy())
                self.u_img_plots[stage].set_data(u.detach().cpu().numpy())
                self.v_img_plots[stage].set_data(v.detach().cpu().numpy())
                # self.s_img_plots[stage].set_data(s.detach().cpu().numpy())
                # self.b_img_plots[stage].set_data(b.detach().cpu().numpy())

    def append_loss(self, loss_h, loss_u, loss_v, loss_s, loss_b):

        # Append to data
        self.h_loss_data = np.append(self.h_loss_data, np.array([loss_h.detach().cpu().numpy()]))
        self.u_loss_data = np.append(self.u_loss_data, np.array([loss_u.detach().cpu().numpy()]))
        self.v_loss_data = np.append(self.v_loss_data, np.array([loss_v.detach().cpu().numpy()]))
        self.s_loss_data = np.append(self.s_loss_data, np.array([loss_s.detach().cpu().numpy()]))
        self.b_loss_data = np.append(self.b_loss_data, np.array([loss_b.detach().cpu().numpy()]))

        # Set X Data
        self.h_loss_plot.set_xdata(range(self.h_loss_data.shape[0]))
        self.u_loss_plot.set_xdata(range(self.u_loss_data.shape[0]))
        self.v_loss_plot.set_xdata(range(self.v_loss_data.shape[0]))
        self.s_loss_plot.set_xdata(range(self.s_loss_data.shape[0]))
        self.b_loss_plot.set_xdata(range(self.b_loss_data.shape[0]))

        # Set Y Data
        self.h_loss_plot.set_ydata(self.h_loss_data)
        self.u_loss_plot.set_ydata(self.u_loss_data)
        self.v_loss_plot.set_ydata(self.v_loss_data)
        self.s_loss_plot.set_ydata(self.s_loss_data)
        self.b_loss_plot.set_ydata(self.b_loss_data)

        # Update the axis limits
        min_y = np.min(np.array([
            np.min(self.h_loss_data),
            np.min(self.u_loss_data),
            np.min(self.v_loss_data),
            np.min(self.s_loss_data),
            np.min(self.b_loss_data),
        ])).item()

        max_y = np.max(np.array([
            np.max(self.h_loss_data),
            np.max(self.u_loss_data),
            np.max(self.v_loss_data),
            np.max(self.s_loss_data),
            np.max(self.b_loss_data),
        ])).item()

        min_max_range = max_y - min_y
        min_y = min_y - 0.1*min_max_range
        max_y = max_y + 0.1*min_max_range

        self.loss_ax.set_ylim([min_y, max_y])



# MATPLOTLIB MULTI-WINDOW
class MultiWindow:

    def __init__(self, width, height):

        self.width = width
        self.height = height

        # Initial blank images
        self.h = torch.zeros(1, 1, height, width)
        self.u = torch.zeros(1, 1, height, width)
        self.v = torch.zeros(1, 1, height, width)
        self.s = torch.zeros(1, 1, height, width)
        self.b = torch.zeros(1, 1, height, width)

        # Matplotlib interactive mode
        plt.ion()

        # Create subplots
        self.figure, self.axs = plt.subplots(2, 3, figsize=(20, 10))

        self.loss_h_plot = self.axs[0, 0].imshow(self.h[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=-0.5, vmax=1)
        self.loss_u_plot = self.axs[0, 1].imshow(self.u[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=-0.5, vmax=1)
        self.loss_v_plot = self.axs[0, 2].imshow(self.v[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=-0.5, vmax=1)
        self.loss_s_plot = self.axs[1, 0].imshow(self.s[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=-0.5, vmax=1)
        self.loss_b_plot = self.axs[1, 1].imshow(self.b[0,0].clone().detach().cpu().numpy(), cmap="gray", vmin=-0.5, vmax=1)

        self.img_alpha = 1

        self.water_plot = self.axs[0, 0].imshow(self.h[0,0].clone().detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=0.02, alpha=self.img_alpha)
        self.momentum_u_plot = self.axs[0, 1].imshow(self.u[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2, alpha=self.img_alpha)
        self.momentum_v_plot = self.axs[0, 2].imshow(self.v[0,0].clone().detach().cpu().numpy(), cmap="bwr", vmin=-0.2, vmax=0.2, alpha=self.img_alpha)
        self.sediment_plot = self.axs[1, 0].imshow(self.s[0,0].clone().detach().cpu().numpy(), cmap="YlOrBr", vmin=0, vmax=0.2, alpha=self.img_alpha)
        self.vegetation_plot = self.axs[1, 1].imshow(self.b[0,0].clone().detach().cpu().numpy(), cmap="YlGn", vmin=0, vmax=1500, alpha=self.img_alpha)

        # Title and axes configuration
        self.axs[0, 0].set(title="Water Layer Thickness", xlabel="Cross shore", ylabel="Along shore")
        self.axs[0, 1].set(title="Momentum u (x-direction)", xlabel="Cross shore", ylabel="Along shore")
        self.axs[0, 2].set(title="Momentum v (y-direction)", xlabel="Cross shore", ylabel="Along shore")
        self.axs[1, 0].set(title="Sediment bed", xlabel="Cross shore", ylabel="Along shore")
        self.axs[1, 1].set(title="Vegetation density", xlabel="Cross shore", ylabel="Along shore")

        # Color bars
        plt.colorbar(self.water_plot)
        plt.colorbar(self.momentum_u_plot)
        plt.colorbar(self.momentum_v_plot)
        plt.colorbar(self.sediment_plot)
        plt.colorbar(self.vegetation_plot)
        
        # Window status
        self.is_open = False

    def set_training_loss(self, training_loss):
        training_loss.plot(ax=self.axs[1, 2])

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

    def set_loss(self, loss_h, loss_u, loss_v, loss_s, loss_b):

        self.loss_h_plot.set_data(loss_h.detach().cpu().numpy())
        self.loss_u_plot.set_data(loss_u.detach().cpu().numpy())
        self.loss_v_plot.set_data(loss_v.detach().cpu().numpy())
        self.loss_s_plot.set_data(loss_s.detach().cpu().numpy())
        self.loss_b_plot.set_data(loss_b.detach().cpu().numpy())






# # OPENCV WINDOW
# class CVWindow:

#     def __init__(self, name, width, height):
#         self.name = name
#         self.padding = 80
#         self.border_size = 2
#         self.width = width + 2*self.padding
#         self.height = height + 2*self.padding
#         self.canvas_width = width
#         self.canvas_height = height

#         # Window image
#         self.img = np.ones((self.height, self.width, 3))

#         cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
#         cv2.setMouseCallback(self.name, self.__mouse_event_handler)
#         cv2.imshow(self.name, self.img)

#         # Canvas image
#         self.data_img = np.zeros((self.canvas_height, self.canvas_width))
#         self.data_min = 0
#         self.data_max = 1
#         self.colormap = colormaps.WHITE_BLUE

#         # Window stats
#         self.open = True
#         self.mouseX = 0
#         self.mouseY = 0
#         self.fps = 0
#         self.fps_tracker = time.time()

#     #
#     # Private methods
#     #
#     def __mouse_event_handler(self, event, x, y, flags, param):
#         self.mouseX = x
#         self.mouseY = y

#     def __is_mouse_in_canvas(self):
#         return self.padding <= self.mouseX < self.width - self.padding and self.padding <= self.mouseY < self.height-self.padding

#     #
#     # Public methods
#     #
#     def draw_text(self, text, x, y):
#         cv2.putText(self.img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

#     def is_open(self):
#         return self.open

#     def put_image(self, img):
#         if not img.shape == self.data_img.shape:
#             raise Exception(f"Expected image of size {self.data_img.shape}, got {img.shape}")

#         self.data_img[:, :] = img[:, :]

#     def get_image(self):
#         return self.data_img.copy()

#     def set_data_range(self, min, max):
#         self.data_min = min
#         self.data_max = max

#     # Main update method
#     def update(self):
#         if not self.open:
#             return

#         # Clear the image
#         self.img[:, :, :] = 255

#         # Canvas border
#         self.img[self.padding-self.border_size:-self.padding+self.border_size, self.padding-self.border_size:-self.padding+self.border_size, :] = 0

#         # Place the canvas image in the center
#         self.img[self.padding:-self.padding, self.padding:-self.padding, :] = self.colormap.transform((self.data_img - self.data_min) / (self.data_max - self.data_min))

#         #
#         # Update fps
#         #
#         t_new = time.time()
#         self.fps = 1.0 / (t_new - self.fps_tracker)
#         self.fps_tracker = t_new
#         self.draw_text(f"{round(self.fps)} FPS", 5, 15)

#         #
#         # Handle keyboard input
#         #
#         key = cv2.waitKey(1)

#         if key == ord('q'):
#             self.open = False

#         # Draw canvas image values (if mouse inside canvas)
#         if self.__is_mouse_in_canvas():
#             self.draw_text(f"Value at ({self.mouseX - self.padding}, {self.mouseY - self.padding}): {round(self.data_img[self.mouseY - self.padding, self.mouseX - self.padding], 5)}", 20, self.height - 20)

#         # Draw the image
#         cv2.imshow(self.name, self.img)




# if __name__ == "__main__":
#     win = CVWindow("test", 1000, 500)

#     colormap = colormaps.BLACK_WHITE

#     img = win.get_image()

#     for x in range(img.shape[1]):
#         for y in range(img.shape[0]):
#             img[y, x] = (x/img.shape[1])*(y/img.shape[0])

#     win.put_image(img)

#     while win.is_open():

#         win.update()