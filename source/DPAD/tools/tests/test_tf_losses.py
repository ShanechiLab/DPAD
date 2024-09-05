""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """

# pylint: disable=C0103, C0111

"Tests tensorflow losses"

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from tools.evaluation import evalPrediction
from tools.tf_losses import (
    computeR2_masked,
    masked_CategoricalCrossentropy,
    masked_CC,
    masked_mse,
    masked_negativeCC,
    masked_negativeR2,
    masked_PoissonLL_loss,
    masked_R2,
    masked_SparseCategoricalCrossentropy,
)
from tools.tools import get_one_hot

# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


def prep_CCE_case(missing_marker=-1):
    # For 4D signals
    np.random.seed(42)

    N, nb, nx, nc = 100, 32, 2, 3
    ZClass = np.random.randint(0, nc, (N, nb, nx))
    Z = get_one_hot(ZClass, nc)
    ZProb = np.random.rand(N, nb, nx, nc)
    ZProb = ZProb / np.sum(ZProb, axis=-1)[..., np.newaxis]
    ZLogProb = np.log(ZProb)
    missing_inds = np.random.randint(0, N, int(np.round(0.4 * N)))
    Z[missing_inds, 0] = missing_marker

    ZR = np.reshape(Z, [np.prod(Z.shape[0:2]), Z.shape[-2], Z.shape[-1]])
    ZLogProbR = np.reshape(
        ZLogProb, [np.prod(ZLogProb.shape[0:2]), ZLogProb.shape[-2], ZLogProb.shape[-1]]
    )

    isOk = np.all(ZR != missing_marker, axis=-1)
    isOk = np.all(isOk, axis=-1)

    ZROk = ZR[isOk, ...]
    ZLogProbROk = ZLogProbR[isOk, ...]

    CCE = -np.mean(np.sum(ZROk * ZLogProbROk, axis=-1))

    return Z, ZClass, ZLogProb, CCE


def assert_params_are_close(s, sBack):
    skipParams = []
    impossibleParams = []
    okParams = []
    errorParams = []
    errorParamsErr = []

    for p, valOrig in s.getListOfParams().items():
        if not hasattr(sBack, p):
            skipParams.append(p)
            continue
        valNew = getattr(sBack, p)
        if valNew is not None:
            try:
                np.testing.assert_allclose(valNew, valOrig, rtol=1e-3, atol=1e-6)
                okParams.append(p)
            except Exception as e:
                errorParams.append(p)
                errorParamsErr.append(e)
                continue
        else:
            impossibleParams.append(p)

    return skipParams, impossibleParams, okParams, errorParams, errorParamsErr


class TestLosses(unittest.TestCase):

    def test_masked_mse_for2D(self):
        # For 2D signals
        np.random.seed(42)

        N, nx = 100, 3
        missing_marker = -1
        Z = np.random.rand(N, nx)
        ZHat = Z + 0.5 * np.random.rand(N, nx)
        missing_inds = np.random.randint(0, N, int(np.round(0.4 * N)))
        Z[missing_inds, 0] = missing_marker

        isOk = np.all(Z != missing_marker, axis=-1)

        expected = np.mean(np.power(Z[isOk, :] - ZHat[isOk, :], 2))

        lossFunc = masked_mse(missing_marker)
        computed = float(lossFunc(Z, ZHat))
        np.testing.assert_allclose(expected, computed, rtol=1e-3)

    def test_masked_mse_for3D(self):
        # For 3D signals
        np.random.seed(42)

        N, nb, nx = 100, 32, 3
        missing_marker = -1
        Z = np.random.rand(N, nb, nx)
        ZHat = Z + 0.5 * np.random.rand(N, nb, nx)
        missing_inds = np.random.randint(0, N, int(np.round(0.4 * N)))
        Z[missing_inds, 0] = missing_marker

        isOk = np.all(Z != missing_marker, axis=-1)

        expected = np.mean(np.power(Z[isOk, ...] - ZHat[isOk, ...], 2))

        lossFunc = masked_mse(missing_marker)
        computed = float(lossFunc(Z, ZHat))
        np.testing.assert_allclose(expected, computed, rtol=1e-3)

    def test_masked_CC_for2D(self):
        # For 2D signals
        np.random.seed(42)

        N, nx = 100, 3
        missing_marker = -1
        Z = np.random.rand(N, nx)
        ZHat = Z + 0.5 * np.random.rand(N, nx)
        missing_inds = np.random.randint(0, N, int(np.round(0.4 * N)))
        Z[missing_inds, 0] = missing_marker

        isOk = np.all(Z != missing_marker, axis=-1)

        expected = np.mean(evalPrediction(Z[isOk, :], ZHat[isOk, :], "CC"))

        lossFunc = masked_CC(missing_marker)
        computed = float(lossFunc(Z, ZHat))
        np.testing.assert_allclose(expected, computed, rtol=1e-3)

        lossFuncNeg = masked_negativeCC(missing_marker)
        computedNeg = float(lossFuncNeg(Z, ZHat))
        np.testing.assert_allclose(-expected, computedNeg, rtol=1e-3)

    def test_masked_R2_for2D(self):
        # For 2D signals
        np.random.seed(42)

        N, nx = 100, 3
        missing_marker = -1
        for test_num in range(10):
            Z = np.random.rand(N, nx)
            ZHat = Z + 0.5 * np.random.rand(N, nx)
            missing_inds = np.random.randint(0, N, int(np.round(0.4 * N)))
            Z[missing_inds, 0] = missing_marker

            flat_chans = int(np.round(np.random.rand()))
            Z[:, :flat_chans] = np.mean(Z[:, :flat_chans], axis=0)

            isOk = np.all(Z != missing_marker, axis=-1)

            allR2_expected = evalPrediction(Z[isOk, :], ZHat[isOk, :], "R2")
            allR2_computed = np.array(
                computeR2_masked(Z, ZHat, missing_marker), dtype=float
            )

            np.testing.assert_allclose(allR2_expected, allR2_computed, rtol=1e-3)

            expected = np.mean(allR2_expected)

            lossFunc = masked_R2(missing_marker)
            computed = float(lossFunc(Z, ZHat))
            np.testing.assert_allclose(expected, computed, rtol=1e-3)

            lossFuncNeg = masked_negativeR2(missing_marker)
            computedNeg = float(lossFuncNeg(Z, ZHat))
            np.testing.assert_allclose(-expected, computedNeg, rtol=1e-3)

    def test_masked_PoissonLL_loss_for3D(self):
        # For 3D signals
        np.random.seed(42)

        N, nb, nx = 100, 32, 3
        missing_marker = -1
        logR = np.random.randn(N, nb, nx)  # Log rates
        R = np.exp(logR)  # Rates
        Z = np.random.poisson(R)  # Counts
        RHat = np.exp(logR + 0.5 * np.random.randn(N, nb, nx))
        missing_inds = np.random.randint(0, N, int(np.round(0.4 * N)))
        Z[missing_inds, 0] = missing_marker

        isOk = np.all(Z != missing_marker, axis=-1)

        ZOk = Z[isOk, ...]
        RHatOk = RHat[isOk, ...]

        # loss = y_pred - y_true * log(y_pred)
        expected = np.mean(RHatOk - ZOk * np.log(RHatOk))

        lossFunc = masked_PoissonLL_loss(missing_marker)
        computed = float(lossFunc(Z, RHat))
        np.testing.assert_allclose(expected, computed, rtol=1e-3)

    def test_masked_CategoricalCrossentropy(self):
        missing_marker = -1
        Z, ZClass, ZLogProb, expected = prep_CCE_case(missing_marker)
        lossFunc = masked_CategoricalCrossentropy(missing_marker)
        computed = float(lossFunc(Z, ZLogProb))
        np.testing.assert_allclose(expected, computed, rtol=1e-3)

    def test_masked_SparseCategoricalCrossentropy(self):
        missing_marker = -1
        Z, ZClass, ZLogProb, expected = prep_CCE_case(missing_marker)
        lossFunc = masked_SparseCategoricalCrossentropy(missing_marker)
        computed = float(lossFunc(ZClass, ZLogProb))
        np.testing.assert_allclose(expected, computed, rtol=1e-2)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
