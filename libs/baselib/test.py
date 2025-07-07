import unittest
import numpy as np
from baselib.utils import sliding_window_1D, sliding_window_1D_overlap, sliding_window_2D


class TestWindow(unittest.TestCase):
    def test_window_1D(self):
        X = np.arange(20)
        width = 3
        result = sliding_window_1D(X, func=np.average, width=width)
        self.assertEqual(result.shape[0], 7)
        result = sliding_window_1D(X, func=np.max, width=width)
        self.assertEqual(result[-1], 19)

    def test_window_1D_equal(self):
        X = np.arange(20)
        width = 5
        result = sliding_window_1D(X, func=np.average, width=width)
        self.assertEqual(result.shape[0], 4)
        result = sliding_window_1D(X, func=np.max, width=width)
        self.assertEqual(result[-1], 19)

    def test_window_1D_overlap(self):
        X = np.arange(20)
        width = 3
        step = 1
        result = sliding_window_1D_overlap(X, func=np.average, width=width, step=step)
        self.assertEqual(result.shape[0], 18)
        result = sliding_window_1D(X, func=np.max, width=width)
        self.assertEqual(result[-1], 19)

    def test_window_1D_self_test(self):
        X = np.arange(20)
        width = 3
        step = 3
        res1 = sliding_window_1D(X, func=np.average, width=width)
        res2 = sliding_window_1D_overlap(X, func=np.average, width=width, step=step)
        self.assertIsInstance(res1, np.ndarray)
        self.assertIsInstance(res2, np.ndarray)
        self.assertListEqual(list(res1), list(res2))


class TestWindow2D(unittest.TestCase):
    def test_window_2D(self):
        X = np.arange(99).reshape(11, 9)
        width = 3
        step = 3
        result = sliding_window_2D(X, func=np.average, width=width, step=step)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 3))


if __name__ == '__main__':
    unittest.main()
