import numpy as np
import math
from .base import LibraryBase, ForwardMixin


class FFTFilter(ForwardMixin, LibraryBase):
    """
    FFT-based filter for a signal.
    Using rfft for simpler analysis of real signals.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sig = np.sin(t)

    >>> fft = FFTFilter()
    >>> freq = fft.frequencies(sig)
    >>> print(freq.shape)  # (129,)  - X.shape[0] // 2 + 1
    >>> # example freq ranges from 0 to 0.5
    >>> filter = np.ones(freq.shape)
    >>> filter[freq < 0.1] = 0
    >>> filter[freq > 0.3] = 4
    >>> filtered = fft(sig, filter)
    >>> plt.plot(sig)
    >>> plt.plot(res)
    """

    def __init__(self, window: float = 1, filter: np.ndarray = None, save_fft: bool = False):
        """
        Args:
            window (float): sample spacing,  1/sampling rate
            filter (np.ndarray): filter to apply to the FFT
            save_fft (bool): save the FFT for later use
        """
        self.window = window  # sample spacing,  1/sampling rate
        self.filter = filter  # filter to apply to the FFT
        self.save_fft = save_fft  # save the FFT for later use
        self.val = None

    def frequencies(self, X):
        """Calculate the frequencies for a signal. Used to create filter from the frequencies.

        Args:
            X (np.ndarray): input signal
        Returns:
            freq (np.ndarray): frequencies
        """
        assert X.ndim == 1, "Input must be 1D"
        N = pow(2, math.ceil(math.log(X.shape[0])/math.log(2)))
        freq = np.fft.rfftfreq(N, self.window)
        return freq

    def forward(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """ Filter a signal using FFT. Calculates FFT for signal, multiplies by filter and return the inverse FFT.

        Args:
            X (np.ndarray): input signal. 1D array.
            y (np.ndarray): filter
        Returns:
            filtered_rest (np.ndarray): filtered signal
        """
        assert X.ndim == 1, "Input must be 1D"
        if y is None:
            print("No filter (y) specified, returning input.")
            return X
        N = pow(2, math.ceil(math.log(X.shape[0])/math.log(2)))
        val = np.fft.rfft(X, N)
        if self.save_fft:
            self.val = val
        assert val.shape[0] == y.shape[0]
        val *= y
        filtered_rest = np.fft.irfft(val)
        return filtered_rest.real
