import numpy as np
import matplotlib as mpl
from .base import BaseSpectralProcessor


def minmaxsacale(data):
    return (data - data.min()) / (data.max() - data.min())


# define color pallets. name: [colors, color values, method]
color_pallets = {"ndvi": [["red", "#FF7441", "#FFE51E", "#FFFDA9", "#00B61E", "#006837"], [0, 45, 64, 105, 200, 255], "interpolation"]}


class Painter(BaseSpectralProcessor):
    def __init__(self, colors=None, color_values=None, method=None):
        self.avalable_methods = ["interpolation", "gaussian"]
        self.method = method
        if self.method is None:
            self.method = "interpolation"  # default method if none is given
        self.colors = colors
        self.color_values = color_values

        if self.colors is not None:
            self.colors = np.array(self.colors)
        else:
            print("No colors provided, exiting Painter")

        if color_values is None and self.colors is not None:
            self.color_values = np.append(np.arange(0, 1, 1 / (len(self.colors) - 1)), 1)
        if self.colors is not None and color_values is not None:
            self.color_values = np.array(self.color_values)
            assert self.colors.shape[0] == self.color_values.shape[0], "number of colors and color values do not match"
            self.color_values = np.sort(self.color_values)
            # scale color gradient values to 0 - 1
            self.color_values = minmaxsacale(self.color_values)

        # default gaussian values
        self.max_width = 1
        self.gradient_resolution = 100

    def _interpolate(self, data):
        color_data = mpl.colors.to_rgba_array(self.colors)[:, :3]
        painted_data = np.zeros(tuple([*data.shape, 3]))
        for i in range(3):
            painted_data[..., i] = np.interp(data, self.color_values, color_data[:, i])
        return painted_data

    def _gauss_color(self, x, a, b, c, d=0):
        return a * np.exp(-np.power((x - b), 2) / (2 * np.power(c, 2))) + d

    def _gauss_pixel_map(self, map, minr=0, maxr=1, spread=1):
        line = np.linspace(minr, maxr, self.gradient_resolution)
        width = float(self.max_width)
        painted_data = np.zeros(tuple([*line.shape, 3]))
        for i in range(3):
            for p in map:
                painted_data[..., i] += self._gauss_color(line, p[1][i], p[0] * width, width / (spread * len(map)))
        return painted_data, line

    def _gaussian(self, data):
        color_data = mpl.colors.to_rgba_array(self.colors)[:, :3]
        heatmap = [[self.color_values[i], tuple(list(color_data[i, :]))] for i in range(self.color_values.shape[0])]
        pixel_map, line = self._gauss_pixel_map(heatmap, data.min(), data.max())
        painted_data = np.zeros(tuple([*data.shape, 3]))
        for i in range(3):
            painted_data[..., i] = np.interp(data, line, pixel_map[:, i])
        painted_data[painted_data > 1] = 1
        return painted_data

    def process(self, data):
        """
        Colorize/paint input data using set method and colors

        Parameters
            data : numpy ndarray
                grayscale data array of dimension 1 or 2. size [pixels] or [x, y]
        """
        if len(data.shape) == 3:
            data = np.squeeze(data)
        assert len(data.shape) == 1 or len(data.shape) == 2, f"given data has too high of a dimension, data dimensions: {data.shape}"
        assert self.method in self.avalable_methods, "unknown method selected"
        if self.method == "interpolation":
            return self._interpolate(data)
        if self.method == "gaussian":
            return self._gaussian(data)

