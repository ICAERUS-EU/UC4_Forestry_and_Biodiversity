import numpy as np
from .base import LibraryBase, ForwardMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer


class VectorNorm(ForwardMixin, LibraryBase):
    """ Compute the vector norm.

    """

    def __init__(self, norm=None, axis=None):
        """
        Args:
            norm (int): The norm to use to normalize each non zero sample.
            axis (int): axis along which to compute the vector norm.

        """
        self.norm = norm
        self.axis = axis

    def forward(self, x):
        """
        Args:
            x (np.ndarray): The input array.
        Returns:
            np.ndarray: The vector norm.
        """
        return np.linalg.norm(x, ord=self.norm, axis=self.axis)


class Scaler(ForwardMixin, LibraryBase):
    """
    Scales the data to have 0 mean and unit variance.

    Attributes:
        scalers (list): List of available scalers.
    """
    scalers = [
        "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
        "Normalizer"
    ]

    def __init__(self, scaler, **kwargs):
        """
        Args:
            scaler (str): Name of the scaler. Chosen from the list of available scalers.
            **kwargs: Keyword arguments for the scaler.
        """

        self.scaler = scaler
        assert self.scaler in self.scalers, "Error selected sclaer is not available"
        if self.scaler == "StandardScaler":
            self.func = StandardScaler(**kwargs)
        elif self.scaler == "MinMaxScaler":
            self.func = MinMaxScaler(**kwargs)
        elif self.scaler == "MaxAbsScaler":
            self.func = MaxAbsScaler(**kwargs)
        elif self.scaler == "RobustScaler":
            self.func = RobustScaler(**kwargs)
        elif self.scaler == "Normalizer":
            self.func = Normalizer(**kwargs)

    def forward(self, X):
        """
        Forward pass of the scaler.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled data.
        """
        X = self.func.fit_transform(X)
        return X


class Amplitude(ForwardMixin, LibraryBase):
    """
    Amplitude normalization.


    """

    def __init__(self, axis=0):
        """
        Args:
            axis (int): Axis to normalize.
        """
        self.axis = axis

    def forward(self, X, y=None, **kwargs):
        """
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Normalized data.
        """
        return np.abs((np.min(X, axis=self.axis)) - np.max(X, axis=self.axis))
