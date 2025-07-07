import numpy as np
import matplotlib as mpl
from baselib.base import ProcessorMixin
from baselib.analysis import Scaler
from .utils import wavelength_to_band, SPECIM_WAVELENGTHS_2X1

# define color pallets. name: [colors, color values, method]
color_pallets = {"ndvi": [["red", "#FF7441", "#FFE51E", "#FFFDA9", "#00B61E", "#006837"], [0, 45, 64, 105, 200, 255], "interpolation"]}


class CCI(ProcessorMixin):
    """
    CCI index - Chlorophyll Carotenoid index

    b1 = 532
    b2 = 630

    y: np.array of wavelenghts if non standart are used

    Value ranges ~ [-0.5; 0.5]
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(532, wls=y)
        b2 = wavelength_to_band(630, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        cci = (X[..., b1] - X[..., b2]) / (X[..., b1] + X[..., b2])
        cci[X[..., b1] + X[..., b2] == 0] = 0
        return cci


class ARI1(ProcessorMixin):
    """
    ARI - Anthocyanine Reflectance index

    b1 = 550
    b2 = 700

    y: np.array of wavelenghts if non standart are used

    Value ranges - [0; 0.2]
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(550, wls=y)
        b2 = wavelength_to_band(700, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        ari = 1/X[..., b1] - 1/X[..., b2]

        ari[X[..., b1] == 0] = 0
        ari[X[..., b2] == 0] = 0
        return ari


class CRI1(ProcessorMixin):
    """
    CRI - Carotenoid reflectance index

    b1 = 510
    b2 = 550

    y: np.array of wavelenghts if non standart are used

    Value ranges - [1; 12]
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(510, wls=y)
        b2 = wavelength_to_band(550, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        cri = 1/X[..., b1] - 1/X[..., b2]

        cri[X[..., b1] == 0] = 0
        cri[X[..., b2] == 0] = 0
        return cri


class SIPI(ProcessorMixin):
    """
    SIPI - Structure intensive pigmenti index

    b1 = 800
    b2 = 445
    b3 = 680

    y: np.array of wavelenghts if non standart are used

    Value range - [0; 2]
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(800, wls=y)
        b2 = wavelength_to_band(445, wls=y)
        b3 = wavelength_to_band(680, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        sipi = (X[..., b1] - X[..., b2]) / (X[..., b1] - X[..., b3])

        sipi[X[..., b1] - X[..., b3] == 0] = 0
        return sipi


class PRI(ProcessorMixin):
    """
    PRI - Photochemical Reflectance index 

    b1 = 531
    b2 = 570

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(531, wls=y)
        b2 = wavelength_to_band(570, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        pri = (X[..., b1] - X[..., b2]) / (X[..., b1] + X[..., b2])

        pri[X[..., b1] + X[..., b2] == 0] = 0
        return pri


class WBI(ProcessorMixin):
    """
    Water Band index

    b1 = 970
    b2 = 900

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(970, wls=y)
        b2 = wavelength_to_band(900, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        wbi = X[..., b1] / X[..., b2]

        wbi[X[..., b2] == 0] = 0
        return wbi


class VRI1(ProcessorMixin):
    """
    VRI - Vogelmann red edge index

    b1 = 740
    b2 = 720

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(740, wls=y)
        b2 = wavelength_to_band(720, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        vri = X[..., b1] / X[..., b2]

        vri[X[..., b2] == 0] = 0
        return vri


class PSSR(ProcessorMixin):
    """
    Pigment specific simple ratio

    b1 = 800
    b2 = 680 - cla
    b3 = 635 - clb
    b4 = 470 - carrotenoid

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(800, wls=y)
        b2 = wavelength_to_band(680, wls=y)
        b3 = wavelength_to_band(635, wls=y)
        b4 = wavelength_to_band(470, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        cla = X[..., b1] / X[..., b2]
        clb = X[..., b1] / X[..., b3]
        car = X[..., b1] / X[..., b4]

        cla[X[..., b2] == 0] = 0
        clb[X[..., b3] == 0] = 0
        car[X[..., b4] == 0] = 0
        return cla, clb, car


class REP(ProcessorMixin):
    """
    Red Edge possition index

    b1 = 700
    b2 = 670
    b3 = 780
    b4 = 740

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(700, wls=y)
        b2 = wavelength_to_band(670, wls=y)
        b3 = wavelength_to_band(780, wls=y)
        b4 = wavelength_to_band(740, wls=y)
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        rep = 700 + 40 * (((X[..., b2] + X[..., b3]) / 2 - X[..., b1]) / (X[..., b4] - X[..., b1]))

        rep = np.nan_to_num(rep)
        rep[rep > 10000] = 0
        rep[rep < -10000] = 0
        return rep


class YI(ProcessorMixin):
    """
    Yellowness index

    b1 = 580
    b2 = 624
    b3 = 668
    l = 0.044

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        b1 = wavelength_to_band(580, wls=y)
        b2 = wavelength_to_band(624, wls=y)
        b3 = wavelength_to_band(668, wls=y)
        l = 0.044
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data needs to be 2D or 3D"
        if y is None:
            assert X.shape[-1] == len(SPECIM_WAVELENGTHS_2X1)
        else:
            assert X.shape[-1] == len(y)

        yi = (X[..., b1] - 2 * X[..., b2] + X[..., b3]) / (l**2)
        return yi


class GRSAI(ProcessorMixin):
    """

    y: np.array of wavelenghts if non standart are used

    Value range - ?
    """

    def process(self, X, y=None):
        pass


class Painter(ProcessorMixin):
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
            scl = Scaler("MinMaxScaler")
            self.color_values = scl(self.color_values.reshape(1, -1)).squeeze()

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

    def process(self, X, y=None):
        """
        Colorize/paint input data using set method and colors

        Parameters
            data : numpy ndarray
                grayscale data array of dimension 1 or 2. size [pixels] or [x, y]
        """
        if len(X.shape) == 3:
            X = np.squeeze(X)
        assert len(X.shape) == 1 or len(X.shape) == 2, f"given data has too high of a dimension, data dimensions: {X.shape}"
        assert self.method in self.avalable_methods, "unknown method selected"
        if self.method == "interpolation":
            return self._interpolate(X)
        if self.method == "gaussian":
            return self._gaussian(X)
