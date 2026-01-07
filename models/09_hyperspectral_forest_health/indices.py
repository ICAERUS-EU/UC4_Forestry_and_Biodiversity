import numpy as np
import os
from utils import ProcessorMixin, wavelength_to_band

BASE_DIR = os.path.dirname(__file__)

# Load wavelengths.npy from the same folder
SPECIM_WAVELENGTHS_2X1 = np.load(os.path.join(BASE_DIR, "wavelengths.npy"))

class CCI(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(532, y)
        b2 = wavelength_to_band(630, y)
        return (X[..., b1] - X[..., b2]) / (X[..., b1] + X[..., b2])


class ARI1(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(550, y)
        b2 = wavelength_to_band(700, y)
        return 1 / X[..., b1] - 1 / X[..., b2]


class CRI1(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(510, y)
        b2 = wavelength_to_band(550, y)
        return 1 / X[..., b1] - 1 / X[..., b2]


class SIPI(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(800, y)
        b2 = wavelength_to_band(445, y)
        b3 = wavelength_to_band(680, y)
        return (X[..., b1] - X[..., b2]) / (X[..., b1] - X[..., b3])


class PRI(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(531, y)
        b2 = wavelength_to_band(570, y)
        return (X[..., b1] - X[..., b2]) / (X[..., b1] + X[..., b2])


class WBI(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(970, y)
        b2 = wavelength_to_band(900, y)
        return X[..., b1] / X[..., b2]


class VRI1(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(740, y)
        b2 = wavelength_to_band(720, y)
        return X[..., b1] / X[..., b2]


class PSSR(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(800, y)
        b2 = wavelength_to_band(680, y)
        b3 = wavelength_to_band(635, y)
        b4 = wavelength_to_band(470, y)

        return (
            X[..., b1] / X[..., b2],
            X[..., b1] / X[..., b3],
            X[..., b1] / X[..., b4],
        )


class REP(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(700, y)
        b2 = wavelength_to_band(670, y)
        b3 = wavelength_to_band(780, y)
        b4 = wavelength_to_band(740, y)

        return 700 + 40 * (
            ((X[..., b2] + X[..., b3]) / 2 - X[..., b1]) /
            (X[..., b4] - X[..., b1])
        )


class YI(ProcessorMixin):
    def process(self, X, y=None):
        b1 = wavelength_to_band(580, y)
        b2 = wavelength_to_band(624, y)
        b3 = wavelength_to_band(668, y)

        return (X[..., b1] - 2 * X[..., b2] + X[..., b3]) / (0.044 ** 2)
