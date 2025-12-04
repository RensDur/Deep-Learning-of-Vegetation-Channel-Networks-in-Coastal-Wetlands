import cv2
import numpy as np
import time
import colormaps

class Window:

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
        self.colormap = colormaps.WHITE_GREEN

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
    win = Window("test", 1000, 500)

    colormap = colormaps.BLACK_WHITE

    img = win.get_image()

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            img[y, x] = (x/img.shape[1])*(y/img.shape[0])

    win.put_image(img)

    while win.is_open():

        win.update()