""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """

# pylint: disable=C0103, C0111

"Tests the module"

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import linalg
from tools.tools import (
    extractDiagonalBlocks,
    getBlockIndsFromBLKSArray,
    isClockwise,
    shortenGaps,
    standardizeStateTrajectory,
)


class TestTools(unittest.TestCase):

    def test_extractDiagonalBlocks(self):
        testCases = [
            [
                {
                    "A": linalg.block_diag(
                        1, 2 * np.ones((3, 3)), 3, 4 * np.ones((2, 2))
                    ),
                },
                np.array([1, 3, 1, 2]),
            ],
            [
                {
                    "A": linalg.block_diag(5 * np.ones((5, 5))),
                },
                np.array([5]),
            ],
            [
                {
                    "A": linalg.block_diag(1 * np.ones((5, 5)), 2 * np.ones((3, 3))),
                },
                np.array([5, 3]),
            ],
            [{"A": np.array([[1, 2], [0, 3]]), "emptySide": "lower"}, np.array([1, 1])],
            [{"A": np.array([[1, 2], [0, 3]]), "emptySide": "upper"}, np.array([2])],
        ]
        for case in testCases:
            input_args = case[0]
            output_correct = case[1]

            BLKS = extractDiagonalBlocks(**input_args)
            np.testing.assert_array_equal(BLKS, output_correct)

    def test_getBlockIndsFromBLKSArray(self):
        testCases = [
            [{"BLKS": [1]}, np.array([[0, 1]])],
            [{"BLKS": [2]}, np.array([[0, 2]])],
            [{"BLKS": [1, 2]}, np.array([[0, 1], [1, 3]])],
            [{"BLKS": [1, 3, 1]}, np.array([[0, 1], [1, 4], [4, 5]])],
        ]
        for case in testCases:
            input_args = case[0]
            output_correct = case[1]

            groups = getBlockIndsFromBLKSArray(**input_args)
            np.testing.assert_array_equal(groups, output_correct)

    def test_shortenGaps(self):
        N = 1000
        T = 100
        t = np.random.rand(N) * T
        t = t[np.argsort(t)][:, np.newaxis]
        timeCopy = copy.copy(t)
        dT = np.median(np.diff(t))

        tNew, timeRemapper = shortenGaps(t)

        timeCopyRemap = timeRemapper.apply(timeCopy)
        np.testing.assert_allclose(tNew, timeCopyRemap)

        t = np.arange(N)
        tNew, timeRemapper = shortenGaps(np.array(t))
        timeCopyRemap = timeRemapper.apply(copy.copy(t))
        np.testing.assert_allclose(tNew, t)
        np.testing.assert_allclose(tNew, timeCopyRemap)

    def test_isClockwise(self):
        np.random.seed(42)

        num_tests = 1000
        for ti in range(num_tests):
            theta0, theta1 = np.sort(np.random.rand(2) * np.pi * 2)
            theta = np.linspace(theta0, theta1, 100)
            c = (np.random.rand(2) - 0.5) * 2
            r = np.random.rand(2)
            x = c[0] + r[0] * np.cos(theta)
            y = c[1] + 10 * r[1] * np.sin(theta)
            data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)

            rotTheta = (np.random.rand(1)[0] - 0.5) * np.pi
            R = np.array(
                [
                    [np.cos(rotTheta), -np.sin(rotTheta)],
                    [np.sin(rotTheta), np.cos(rotTheta)],
                ]
            )
            data = (R @ data.T).T

            obsInds = np.sort(
                np.random.randint(0, theta.size, np.random.randint(5, 30))
            )

            isCW = isClockwise(data[obsInds, :])
            np.testing.assert_equal(isCW, False)

            isCW = isClockwise(np.flipud(data[obsInds, :]))
            np.testing.assert_equal(isCW, True)

    def test_standardizeStateTrajectory(self):
        np.random.seed(42)

        num_tests = 1000
        for ti in range(num_tests):
            theta0, theta1 = np.sort(np.random.rand(2) * np.pi * 2)
            theta = np.linspace(theta0, theta1, 100)
            elev0, elev1 = np.sort(np.random.rand(2) * np.pi * 2)
            elev = np.linspace(elev0, elev1, 100)
            c = (np.random.rand(3) - 0.5) * 2
            r = np.random.rand(3)
            x = c[0] + r[0] * np.cos(theta)
            y = c[1] + 10 * r[1] * np.sin(theta)
            z = c[2] + 5 * r[2] * np.sin(elev)
            data = np.concatenate(
                (x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1
            )

            rotTheta = (np.random.rand(1)[0] - 0.5) * np.pi
            R = np.array(
                [
                    [np.cos(rotTheta), -np.sin(rotTheta)],
                    [np.sin(rotTheta), np.cos(rotTheta)],
                ]
            )
            R1 = np.block([[R, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones(1)]])

            rotElev = (np.random.rand(1)[0] - 0.5) * np.pi
            R = np.array(
                [
                    [np.cos(rotElev), -np.sin(rotElev)],
                    [np.sin(rotElev), np.cos(rotElev)],
                ]
            )
            R2 = np.block([[np.ones(1), np.zeros((1, 2))], [np.zeros((2, 1)), R]])
            data = (R2 @ R1 @ data.T).T

            xMean = np.random.rand(3)
            data = data + xMean

            obsInds = np.unique(
                np.random.randint(0, theta.size, np.random.randint(25, 40))
            )

            for nx in [1, 3]:
                xTest = data[obsInds, :nx]
                xTestN, E, X0 = standardizeStateTrajectory(xTest, generate_plot=False)

                # Outputs should describe the same similarity transform
                xTestNExpected = (E @ (xTest - X0).T).T
                np.testing.assert_allclose(xTestN, xTestNExpected)

                # Zero mean
                np.testing.assert_almost_equal(np.mean(xTestN, axis=0), np.zeros(nx))

                # Start from the positive side of the x-axis (have no y-element in the start)
                np.testing.assert_array_less(0, xTestN[0, 0])
                if nx > 1:
                    np.testing.assert_almost_equal(xTestN[0, 1], 0)

                    # Be counter clockwise on the xy-plane
                    isCW = isClockwise(xTestN[:, :2])
                    np.testing.assert_equal(isCW, False)


if __name__ == "__main__":
    unittest.main()
