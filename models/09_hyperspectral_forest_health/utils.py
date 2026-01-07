import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

# Load wavelengths.npy from the same folder
SPECIM_WAVELENGTHS_2X1 = np.load(os.path.join(BASE_DIR, "wavelengths.npy"))

def wavelength_to_band(wl, wls=None):
    if wls is None:
        wls = SPECIM_WAVELENGTHS_2X1
    wls = np.array(wls)
    return np.argmin(np.abs(wls - wl))


class ProcessorMixin:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **params):
        return self.process(X, y=y, **params)

    def forward(self, X, y=None, **params):
        return self.process(X, y=y, **params)

    def __call__(self, X, y=None, **params):
        return self.forward(X, y=y, **params)
