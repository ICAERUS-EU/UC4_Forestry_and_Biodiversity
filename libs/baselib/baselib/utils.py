import numpy as np
import math
from .base import Mixin_function_dict 


def sliding_window_1D(X, func, *, width: int = 3) -> np.ndarray:
    """Apply a function to a sliding window of a 1D array without overlap.

    Args:
        X (np.ndarray): 1D array
        func (function): function to apply to the sliding window
        width (int): width of the sliding window
    Returns:
        np.ndarray: 1D array of the results of the function
    """
    assert width > 1, "width must be greater than 1. otherwise use the function"
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert isinstance(width, int), "width must be an integer"
    assert X.ndim == 1

    if X.shape[0] % width != 0:
        size = X.shape[0] // width
        end = size * width
        vals = np.split(X[:end], size)
        res = np.zeros(size+1)
        for i in range(size):
            res[i] = func(vals[i])
        res[-1] = func(X[end:])
        return res
    else:
        size = X.shape[0] // width
        vals = np.split(X, size)
        res = np.zeros(size)
        for i in range(size):
            res[i] = func(vals[i])
        return res


def sliding_window_1D_overlap(X, func, *, width: int = 3, step: int = 1) -> np.ndarray:
    """Apply a function to a sliding window of a 1D array with overlap.

    Args:
        X (np.ndarray): 1D array
        func (function): function to apply to the sliding window
        width (int): width of the sliding window
        step (int): step of the sliding window
    Returns:
        np.ndarray: array of the results
    """
    assert width > 1, "width must be greater than 1. otherwise use the function"
    assert step > 0
    assert X.ndim == 1

    i = 0
    res = []
    while i * step + width <= X.shape[0]:
        res.append(func(X[i * step: i * step + width]))
        i += 1
    end = (i-1) * step + width
    if end < X.shape[0]:
        res.append(func(X[end:]))
    return np.array(res)


def sliding_window_2D(X, func, *, width: int | list = 3, step: int | list = 1) -> np.ndarray:
    """Apply a function to a sliding window of a 2D array with overlap.

    Args:
        X (np.ndarray): 2D array
        func (function): function to apply to the sliding window
        width (int | list, optional): width of the sliding window. Defaults to 3.
        step (int | list, optional): step size
    Returns:
        np.ndarray: array of results
    """
    assert X.ndim == 2

    if type(width) is int:
        kernel = (width, width)
    elif len(width) == 2:  # can be list or tuple
        kernel = width
    else:
        print("width has to be int or list/tuple of 2 ints")
    if type(step) is int:
        steps = (step, step)
    elif len(step) == 2:  # can be list or tuple
        steps = step
    else:
        print("step has to be int or list/tuple of 2 ints")

    xsteps = math.floor(X.shape[0] / steps[0])
    ysteps = math.floor(X.shape[1] / steps[1])
    res = []
    for x in range(xsteps):
        res_inner = []
        for y in range(ysteps):
            tmp = X[x*steps[0]:x*steps[0]+kernel[0], y*steps[1]:y*steps[1]+kernel[1]]
            res_inner.append(func(tmp))
            #  print(x*steps[0], x*steps[0]+kernel[0], y*steps[1], y*steps[1]+kernel[1])
        if ysteps * steps[1] < X.shape[1]:  # check if data left outside the range on y axis
            tmp = X[xsteps * steps[0]:, ysteps * steps[1]:]
            res_inner.append(func(tmp))
        res.append(res_inner)
    if xsteps * steps[0] < X.shape[0]:  # chek if data left outside the range on x axis
        res_side = []
        for y in range(ysteps):
            tmp = X[xsteps * steps[0]:, y*steps[1]:y*steps[1]+kernel[1]]
            res_side.append(func(tmp))
        res.append(res_side)
    return np.array(res)


def test_class_methods(cls, method_dict=Mixin_function_dict):
    """
    Tests if cls has the required methods implemented
    :param cls: class to test
    :param method_dict: dictionary of methods to test
    :return:
    """
    check = True
    for klass in cls.__mro__:
        if klass.__name__ in method_dict.keys():
            for method in method_dict[klass.__name__]:
                assert hasattr(cls, method), "{} has no required method {} implemented".format(klass.__name__, method)
                if not hasattr(cls, method):
                    check = False
    return check
