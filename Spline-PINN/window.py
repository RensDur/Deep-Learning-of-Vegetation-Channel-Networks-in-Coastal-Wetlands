import cv2
import numpy as np
import time
import colormaps

import torch
import matplotlib.pyplot as plt


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






# OPENCV WINDOW
class CVWindow:

    def __init__(self, name, width, height):
        self.name = name
        self.padding = 80
        self.border_size = 2
        self.width = width + 2*self.padding
        self.height = height + 2*self.padding
        self.canvas_width = width
        self.canvas_height = height

        # Window image
        self.img = np.ones((self.height, self.width, 3))

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.name, self.__mouse_event_handler)
        cv2.imshow(self.name, self.img)

        # Canvas image
        self.data_img = np.zeros((self.canvas_height, self.canvas_width))
        self.data_min = 0
        self.data_max = 1
        self.colormap = colormaps.WHITE_BLUE

        # Window stats
        self.open = True
        self.mouseX = 0
        self.mouseY = 0
        self.fps = 0
        self.fps_tracker = time.time()

    #
    # Private methods
    #
    def __mouse_event_handler(self, event, x, y, flags, param):
        self.mouseX = x
        self.mouseY = y

    def __is_mouse_in_canvas(self):
        return self.padding <= self.mouseX < self.width - self.padding and self.padding <= self.mouseY < self.height-self.padding

    #
    # Public methods
    #
    def draw_text(self, text, x, y):
        cv2.putText(self.img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    def is_open(self):
        return self.open

    def put_image(self, img):
        if not img.shape == self.data_img.shape:
            raise Exception(f"Expected image of size {self.data_img.shape}, got {img.shape}")

        self.data_img[:, :] = img[:, :]

    def get_image(self):
        return self.data_img.copy()

    def set_data_range(self, min, max):
        self.data_min = min
        self.data_max = max

    # Main update method
    def update(self):
        if not self.open:
            return

        # Clear the image
        self.img[:, :, :] = 255

        # Canvas border
        self.img[self.padding-self.border_size:-self.padding+self.border_size, self.padding-self.border_size:-self.padding+self.border_size, :] = 0

        # Place the canvas image in the center
        self.img[self.padding:-self.padding, self.padding:-self.padding, :] = self.colormap.transform((self.data_img - self.data_min) / (self.data_max - self.data_min))

        #
        # Update fps
        #
        t_new = time.time()
        self.fps = 1.0 / (t_new - self.fps_tracker)
        self.fps_tracker = t_new
        self.draw_text(f"{round(self.fps)} FPS", 5, 15)

        #
        # Handle keyboard input
        #
        key = cv2.waitKey(1)

        if key == ord('q'):
            self.open = False

        # Draw canvas image values (if mouse inside canvas)
        if self.__is_mouse_in_canvas():
            self.draw_text(f"Value at ({self.mouseX - self.padding}, {self.mouseY - self.padding}): {round(self.data_img[self.mouseY - self.padding, self.mouseX - self.padding], 5)}", 20, self.height - 20)

        # Draw the image
        cv2.imshow(self.name, self.img)




if __name__ == "__main__":
    win = CVWindow("test", 1000, 500)

    colormap = colormaps.BLACK_WHITE

    img = win.get_image()

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            img[y, x] = (x/img.shape[1])*(y/img.shape[0])

    win.put_image(img)

    while win.is_open():

        win.update()