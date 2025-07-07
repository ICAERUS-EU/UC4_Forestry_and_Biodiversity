import numpy as np
import math
from colour import XYZ_to_RGB
from colour.colorimetry import wavelength_to_XYZ
from .base import LibraryBase, ForwardMixin


class RGB(ForwardMixin, LibraryBase):
    """RGB library. Used to convert hyperspectral data to RGB. Using CIE 1931 standard.
    """

    def __init__(self, wavelength, max_size=200_000_000, normalize=False):
        """
        Args:
            wavelength (array): array of wavelengths in nm.
            max_size (int): maximum size of array to integrate. if size > max_size cut array to pieces. 200Mb limit default
            normalize (bool): normalize data to 0-1 range. Each RGB band will be normalized separately.
        """
        self.wavelength = wavelength  # wavelength numbers in nm.
        self.max_size = max_size   # max size of array to integrate. if size > max_size cut array to pieces. 200Mb limit default
        self.normalize = normalize  # normalize data to 0-1 range. Each RGB band will be normalized separately.

    def forward(self, data):
        """
        Args:
            data (array): 3D hyperspectral data.
        Returns:
            array: RGB data.
        """
        res = hsi_to_xyz(data, self.wavelength, self.max_size, self.normalize)
        return XYZ_to_RGB(res)


def hsi_to_xyz(data, wavelength, max_size, normalize=False):
    """ Convert HSI data to XYZ by integrating HSI values over XYZ color curves.
    Args:
        data (array): 3D hyperspectral data.
        wavelength (array): wavelength numbers in nm.
        max_size (int): maximum size of array to integrate. if size > max

    Returns:
        array: XYZ data.
    """
    sh = data.shape
    bands = get_bands(wavelength)
    if len(data.shape) > 2:
        data = data[..., np.isin(wavelength, bands)].reshape((sh[0] * sh[1], len(bands)))
    else:
        data = data[:, np.isin(wavelength, bands)]

    size = np.prod(data.shape)
    if size > max_size:
        image = None
        for arr in np.array_split(data, math.ceil(size / max_size)):

            if image is None:
                image = integrate(arr, bands)
            else:
                image = np.append(image, integrate(arr, bands), axis=0)
        if normalize:
            image[:, 0] = (image[:, 0] - image[:, 0].min()) / (image[:, 0].max() - image[:, 0].min())
            image[:, 1] = (image[:, 1] - image[:, 1].min()) / (image[:, 1].max() - image[:, 1].min())
            image[:, 2] = (image[:, 2] - image[:, 2].min()) / (image[:, 2].max() - image[:, 2].min())
    else:
        image = integrate(data, bands, normalize)
    if len(sh) > 2:
        image = image.reshape((sh[0], sh[1], 3))
    return image


def x_cmf(band):
    """
    Args:
        band: wavelengths
    Returns:
        array: XYZ color curve for X value
    """
    return wavelength_to_XYZ(band)[:, 0]


def y_cmf(band):
    """
    Args:
        band: wavelengths
    Returns:
        array: XYZ color curve for Y value
    """
    return wavelength_to_XYZ(band)[:, 1]


def z_cmf(band):
    """
    Args:
        band: wavelengths
    Returns:
        array: XYZ color curve for Z value
    """
    return wavelength_to_XYZ(band)[:, 2]


def get_bands(wavelength):
    """ Filter only required spectral bands
    Args:
        wavelength: wavelengths
    Returns:
        array: spectral bands
    """
    return wavelength[np.logical_and(380 < wavelength, wavelength < 780)]


def integrate(data, bands, normalize=False):
    """ Integrate spectral data to XYZ color space
    Args:
        data: spectral data
        bands: spectral bands
        normalize: normalize data
    Returns:
        array: XYZ color space
    """
    X = np.trapz(data * x_cmf(bands), bands.reshape((1, -1)), axis=1)
    if normalize:
        image = (X - X.min()) / (X.max() - X.min())
    else:
        image = X
    X = None

    Y = np.trapz(data * y_cmf(bands) * 1.2, bands.reshape((1, -1)), axis=1)
    if normalize:
        image = np.vstack((image, (Y - Y.min()) / (Y.max() - Y.min())))
    else:
        image = np.vstack((image, Y))
    Y = None

    Z = np.trapz(data * z_cmf(bands), bands.reshape((1, -1)), axis=1)
    if normalize:
        image = np.vstack((image, (Z - Z.min()) / (Z.max() - Z.min())))
    else:
        image = np.vstack((image, Z))
    Z = None
    return image.T
