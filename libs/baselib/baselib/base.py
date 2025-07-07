from sklearn.base import BaseEstimator


# What functions are expected to be implemented by which mixin
Mixin_function_dict = {
    # sklearn mixins
    "BiclusterMixin": ["fit"],
    "ClassNamePrefixFeaturesOutMixin": ["fit"],
    "ClassifierMixin": ["fit", "predict"],
    "ClusterMixin": ["fit"],
    "DensityMixin": ["fit"],
    "MetaEstimatorMixin": ["fit"],
    "OneToOneFeatureMixin": ["fit"],
    "OutlierMixin": ["fit", "predict"],
    "RegressorMixin": ["fit", "predict"],
    "TransformerMixin": ["fit", "transform"],  # fit return self only
    # custom mixins
    "ForwardMixin": ["forward"],
    "ProcessorMixin": ["process"],
}


class LibraryBase(BaseEstimator):
    """
    Base class for all library estimators. This class is not meant to be used directly.
    Implement optional forward function to use the functions the same as pytorch modules are used.

    Extend this class to create a new library estimator.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def forward(self):
        """
        Forward function that is called directly if the module is called.
        Implement this function to use the module like a pytorch module.
        """
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ForwardMixin:
    """
    Mixin class for modules that have a forward function. Runs forward functions in sklearn pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **params):
        return self.forward(X, y=y, **params)


class ProcessorMixin:
    """
    Mixin class for modules that have a process function. Runs process functions in sklearn pipeline.
    For converting process functions to forward and transform functions.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **params):
        return self.process(X, y=y, **params)

    def forward(self, X, y=None, **params):
        return self.process(X, y=y, **params)

    def __call__(self, X, y=None, **params):
        return self.forward(X, y=y, **params)


class ReaderBase:
    """
    Base class for readers.
    """

    def read(self) -> None:
        raise NotImplementedError

    @property
    def data(self):
        return self._data


class WriterBase:
    """
    Base class for writers.
    """

    def write(self) -> None:
        raise NotImplementedError


class LoaderBase:
    """
    Base class for loaders.
    """

    def load(self) -> None:
        raise NotImplementedError
