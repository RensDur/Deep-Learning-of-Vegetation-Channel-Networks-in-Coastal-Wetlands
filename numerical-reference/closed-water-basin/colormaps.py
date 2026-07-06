import math
import numpy as np

class Colormap:

    def __init__(self, xmin, xmax, map):
        self.xmin = xmin
        self.xmax = xmax
        self.map = map

    def sample(self, x):

        if not (x >= self.xmin and x <= self.xmax):
            raise Exception(f"Cannot sample outside colormap boundaries. Received x={x}")

        x = (x - self.xmin) / (self.xmax - self.xmin)
        x *= (self.map.shape[0]-1)

        lowest = math.floor(x)
        highest = math.ceil(x)
        part = (x - lowest)

        # Use linear interpolation to find the corresponding color value
        return (1-part) * self.map[lowest] + part * self.map[highest]

    def transform(self, img):

        resolution = self.map.shape[0]

        # Scale values in the image from [0,1] to [0,resolution-1]
        x = img[:, :] * (resolution-1)
        x = np.clip(x, 0, resolution-1)

        # Find the lower and upper indices
        x0 = np.floor(x).astype(int)
        x1 = np.clip(x0+1, 0, resolution-1)

        # Find the fractional part
        frac = x - x0

        # Use linear interpolation to color the image
        rgb = (1.0 - frac)[..., None] * self.map[x0] + frac[..., None] * self.map[x1]

        return rgb


#
# Build all the colormaps
#


# Black-white
resolution = 50
cmap = np.array([i/resolution for i in range(resolution)]).repeat(3).reshape(resolution, 3)
BLACK_WHITE = Colormap(0, 1, cmap)

red_cmap = cmap.copy()
red_cmap[:, 0:2] = 0
BLACK_RED = Colormap(0, 1, red_cmap)

green_cmap = cmap.copy()
green_cmap[:, 0] = 0
green_cmap[:, 2] = 0
BLACK_GREEN = Colormap(0, 1, green_cmap)

blue_cmap = cmap.copy()
blue_cmap[:, 1:3] = 0
BLACK_BLUE = Colormap(0, 1, blue_cmap)

white_red_cmap = cmap.copy()
white_red_cmap[:, 0:2] = 1 - white_red_cmap[:, 0:2]
white_red_cmap[:, 2] = 1
WHITE_RED = Colormap(0, 1, white_red_cmap)

white_green_cmap = cmap.copy()
white_green_cmap[:, 0] = 1 - white_green_cmap[:, 0]
white_green_cmap[:, 2] = 1 - white_green_cmap[:, 2]
white_green_cmap[:, 1] = 1
WHITE_GREEN = Colormap(0, 1, white_green_cmap)

white_blue_cmap = cmap.copy()
white_blue_cmap[:, 1:3] = 1 - white_blue_cmap[:, 1:3]
white_blue_cmap[:, 0] = 1

resolution = 25
blue_black = np.array([(0.2 + 0.8*(i/resolution)) for i in reversed(range(resolution))])
white_blue_cmap[-resolution:, 0] = blue_black
WHITE_BLUE = Colormap(0, 1, white_blue_cmap)