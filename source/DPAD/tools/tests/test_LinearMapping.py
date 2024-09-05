""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """

# pylint: disable=C0103, C0111

"Tests LinearMapping"

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from ..LinearMapping import LinearMapping

# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


class TestLinearMapping(unittest.TestCase):
    def test_LinearMapping(self):
        np.random.seed(42)

        numTests = 100
        for ci in range(numTests):
            ny = np.random.choice(np.arange(1, 10))
            n_rem = np.random.choice(np.arange(0, ny))

            N = 100
            Y = np.random.randn(N, ny)
            rem_inds = np.unique(np.random.random_integers(0, ny - 1, n_rem))
            keep_vector = np.array([yi not in rem_inds for yi in range(ny)])
            Y[:, rem_inds] = 0

            LM = LinearMapping()
            LM.set_to_dimension_remover(keep_vector)
            Y_rem = LM.apply(Y.T).T
            Y_recover = LM.apply_inverse(Y_rem.T).T

            np.testing.assert_array_equal(Y_rem, Y[:, keep_vector])
            np.testing.assert_array_equal(Y_recover, Y)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
