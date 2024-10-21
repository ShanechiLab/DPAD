""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Tests DPAD"""

# pylint: disable=C0103, C0111

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from DPAD.DPADModel import DPADModel, shift_ms_to_1s_series
from DPAD.sim import (
    generateRandomLinearModel,
    genRandomGaussianNoise,
    getSysSettingsFromSysCode,
)
from DPAD.tools.file_tools import pickle_load, pickle_save

numTests = 10


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


def createDPADModelFromLSSM(s, DPADModelArgs, setToLSSMArgs):
    try:
        s2 = DPADModel(**DPADModelArgs)
        E = s2.setToLSSM(s, **setToLSSMArgs)
    except Exception as e:
        print(e)
        s2 = DPADModel(n1=s.zDims.size + 1, nx=s.state_dim, **DPADModelArgs)
        E = s2.setToLSSM(s, **setToLSSMArgs)
    return s2, E


def getCorrectShiftedVersions(X, steps=range(1, 20)):
    X_step = {}
    for step in steps:
        if step == 1:
            X_step[step] = X
        else:
            if len(X.shape) == 2:
                X_step[step] = np.concatenate(
                    (np.nan * np.ones((step - 1, X.shape[1])), X[: (X.shape[0] - 1), :])
                )[: X.shape[0], ...]
            elif len(X.shape) == 3:
                X_step[step] = np.concatenate(
                    (
                        np.nan * np.ones((step - 1, X.shape[1], X.shape[2])),
                        X[: (X.shape[0] - 1), ...],
                    )
                )[: X.shape[0], ...]
    return X_step


class TestDPAD(unittest.TestCase):
    def test_shift_ms_to_1s_series(self):
        Y = np.array(
            [
                0.1 + np.arange(1, 21, 1),
                0.2 + np.arange(1, 21, 1),
                0.3 + np.arange(1, 21, 1),
            ]
        ).T
        Z = np.array(
            [
                0.5 + np.arange(1, 21, 1),
                0.6 + np.arange(1, 21, 1),
                0.7 + np.arange(1, 21, 1),
            ]
        ).T
        YZ = np.concatenate((Y[..., np.newaxis], Z[..., np.newaxis]), axis=2)
        Y_step = {}
        YZ_step = {}
        for step in range(1, 20):
            if step == 1:
                Y_step[step] = Y
                YZ_step[step] = YZ
            else:
                Y_step[step] = np.concatenate(
                    (np.nan * np.ones((step - 1, Y.shape[1])), Y[: (Y.shape[0] - 1), :])
                )[: Y.shape[0], ...]
                YZ_step[step] = np.concatenate(
                    (
                        np.nan * np.ones((step - 1, YZ.shape[1], YZ.shape[2])),
                        YZ[: (YZ.shape[0] - 1), ...],
                    )
                )[: YZ.shape[0], ...]

        testCases = [
            [{"Y": None, "steps_ahead": [1, 2, 3]}, None],  # Input  # Output
            [{"Y": Y, "steps_ahead": None}, (Y,)],  # Input  # Output
            [{"Y": Y, "steps_ahead": [2]}, (Y_step[2],)],  # Input  # Output
            [
                {"Y": Y, "steps_ahead": [2, 5, 10, 7, 1]},  # Input
                tuple([Y_step[step] for step in [2, 5, 10, 7, 1]]),  # Output
            ],
            [
                {
                    "Y": [Y for _ in range(len([2, 5, 10, 7, 1]))],
                    "steps_ahead": [2, 5, 10, 7, 1],
                },  # Input
                tuple([Y_step[step] for step in [2, 5, 10, 7, 1]]),  # Output
            ],
            [
                {
                    "Y": [YZ for _ in range(len([2, 5, 10, 7, 1]))],
                    "steps_ahead": [2, 5, 10, 7, 1],
                },  # Input
                tuple([YZ_step[step] for step in [2, 5, 10, 7, 1]]),  # Output
            ],
            [
                {
                    "Y": Y.T,
                    "steps_ahead": [2, 5, 10, 7, 1],
                    "time_first": False,
                },  # Input
                tuple([Y_step[step].T for step in [2, 5, 10, 7, 1]]),  # Output
            ],
            [
                {
                    "Y": [Y_step[step].T for step in [2, 5, 1]],
                    "steps_ahead": [2, 3, 2],
                    "time_first": False,
                },  # Input
                tuple([Y_step[step].T for step in [3, 7, 2]]),  # Output
            ],
            [
                {
                    "Y": [YZ_step[step].transpose(1, 0, 2) for step in [2, 5, 1]],
                    "steps_ahead": [2, 3, 2],
                    "time_first": False,
                },  # Input
                tuple(
                    [YZ_step[step].transpose(1, 0, 2) for step in [3, 7, 2]]
                ),  # Output
            ],
        ]
        for caseInd, case in enumerate(testCases):
            input_arg = case[0]
            output_correct = case[1]
            with self.subTest(caseInd=caseInd, input_arg=input_arg):
                output = shift_ms_to_1s_series(**input_arg)
                np.testing.assert_equal(output, output_correct)

    def test_DPADModel_prepare_inputs_to_model1_Cy(self):
        Y = np.array(
            [
                0.10 + np.arange(1, 21, 1),
                0.11 + np.arange(1, 21, 1),
                0.12 + np.arange(1, 21, 1),
            ]
        ).T
        Z = np.array(
            [
                0.20 + np.arange(1, 21, 1),
                0.21 + np.arange(1, 21, 1),
                0.22 + np.arange(1, 21, 1),
            ]
        ).T
        U = np.array(
            [
                0.80 + np.arange(1, 21, 1),
                0.81 + np.arange(1, 21, 1),
                0.82 + np.arange(1, 21, 1),
            ]
        ).T
        X = np.array(
            [
                0.90 + np.arange(1, 21, 1),
                0.91 + np.arange(1, 21, 1),
                0.92 + np.arange(1, 21, 1),
            ]
        ).T
        YZ = np.concatenate((Y, Z), axis=1)
        XU = np.concatenate((X, U), axis=1)
        Y_step = getCorrectShiftedVersions(Y)
        Z_step = getCorrectShiftedVersions(Z)
        U_step = getCorrectShiftedVersions(U)
        X_step = getCorrectShiftedVersions(X)
        YZ_step = getCorrectShiftedVersions(YZ)
        XU_step = getCorrectShiftedVersions(XU)

        testCases = [
            # Cases without input
            [
                {
                    "Y": None,
                    "U": None,
                    "allXp1_steps": None,
                    "steps_ahead": [1, 2, 3],
                },  # Input
                (None, None, None),  # Output
                {"nu": 3, "n1": 3, "nx": 5, "missing_marker": np.nan},  # DPADModelArgs
            ],
            [
                {
                    "Y": Y.T,
                    "U": None,
                    "allXp1_steps": [X_step[s].T for s in [1]],
                    "steps_ahead": [1],
                },  # Input
                (
                    [
                        X_step[s].T for s in [1]
                    ],  # Cat of allXp1_steps and appropriately shifted U
                    [
                        X_step[s].T for s in [1]
                    ],  # Previous output, shifted to time step of 1-step
                    [
                        Y.T for _ in [1]
                    ],  # Repetition of Y output with the original timestep
                ),  # Output
                {"nu": 0, "n1": 3, "nx": 5, "missing_marker": np.nan},  # DPADModelArgs
            ],
            [
                {
                    "Y": Y.T,
                    "U": None,
                    "allXp1_steps": [X_step[s].T for s in [1, 1, 1]],
                    "steps_ahead": [1, 2, 3],
                },  # Input
                (
                    [
                        X_step[s].T for s in [1, 1, 1]
                    ],  # Cat of allXp1_steps and appropriately shifted U
                    [
                        X_step[s].T for s in [1, 2, 3]
                    ],  # Previous output, shifted to time step of 1-step
                    [
                        Y.T for _ in [1, 2, 3]
                    ],  # Repetition of Y output with the original timestep
                ),  # Output
                {"nu": 0, "n1": 3, "nx": 5, "missing_marker": np.nan},  # DPADModelArgs
            ],
            [
                {
                    "Y": Y.T,
                    "U": None,
                    "allXp1_steps": [X_step[s].T for s in [1, 4]],
                    "steps_ahead": [1, 5],
                },  # Input
                (
                    [
                        X_step[s].T for s in [1, 4]
                    ],  # Cat of allXp1_steps and appropriately shifted U
                    [
                        X_step[s].T for s in [1, 8]
                    ],  # Previous output, shifted to time step of 1-step
                    [
                        Y.T for _ in [1, 2]
                    ],  # Repetition of Y output with the original timestep
                ),  # Output
                {"nu": 0, "n1": 3, "nx": 5, "missing_marker": np.nan},  # DPADModelArgs
            ],
            # Cases with input
            [
                {
                    "Y": Y.T,
                    "U": U.T,
                    "allXp1_steps": [X_step[s].T for s in [1]],
                    "steps_ahead": [1],
                },  # Input
                (
                    [
                        XU_step[s].T for s in [1]
                    ],  # Cat of allXp1_steps and appropriately shifted U
                    [
                        XU_step[s].T for s in [1]
                    ],  # Previous output, shifted to time step of 1-step
                    [
                        Y.T for _ in [1]
                    ],  # Repetition of Y output with the original timestep
                ),  # Output
                {
                    "nu": U.shape[-1],
                    "n1": 3,
                    "nx": 5,
                    "missing_marker": np.nan,
                },  # DPADModelArgs
            ],
        ]
        for caseInd, case in enumerate(testCases):
            input_arg = case[0]
            output_correct = case[1]
            DPADModelArgs = case[2]
            with self.subTest(caseInd=caseInd, input_arg=input_arg):
                s = DPADModel(**DPADModelArgs)
                output = s.prepare_inputs_to_model1_Cy(**input_arg)
                np.testing.assert_equal(output, output_correct)

    def test_DPADModel_prepare_inputs_to_model2_Cz(self):
        Y = np.array(
            [
                0.10 + np.arange(1, 21, 1),
                0.11 + np.arange(1, 21, 1),
                0.12 + np.arange(1, 21, 1),
            ]
        ).T
        Z = np.array(
            [
                0.20 + np.arange(1, 21, 1),
                0.21 + np.arange(1, 21, 1),
                0.22 + np.arange(1, 21, 1),
            ]
        ).T
        Z2 = np.array(
            [
                0.30 + np.arange(1, 21, 1),
                0.31 + np.arange(1, 21, 1),
                0.32 + np.arange(1, 21, 1),
            ]
        ).T
        U = np.array(
            [
                0.80 + np.arange(1, 21, 1),
                0.81 + np.arange(1, 21, 1),
                0.82 + np.arange(1, 21, 1),
            ]
        ).T
        X = np.array(
            [
                0.90 + np.arange(1, 21, 1),
                0.91 + np.arange(1, 21, 1),
                0.92 + np.arange(1, 21, 1),
            ]
        ).T
        X2 = np.array(
            [
                0.60 + np.arange(1, 21, 1),
                0.61 + np.arange(1, 21, 1),
                0.62 + np.arange(1, 21, 1),
            ]
        ).T
        YZ = np.concatenate((Y, Z), axis=1)
        XU = np.concatenate((X, U), axis=1)
        Y_step = getCorrectShiftedVersions(Y)
        Z_step = getCorrectShiftedVersions(Z)
        Z2_step = getCorrectShiftedVersions(Z2)
        U_step = getCorrectShiftedVersions(U)
        X_step = getCorrectShiftedVersions(X)
        X2_step = getCorrectShiftedVersions(X2)
        YZ_step = getCorrectShiftedVersions(YZ)
        XU_step = getCorrectShiftedVersions(XU)

        testCases = [
            # Cases without input
            [
                {
                    "Y": None,
                    "Z": None,
                    "U": None,
                    "allX_steps": None,
                    "allXp2_steps": None,
                    "allZp_steps": None,
                    "steps_ahead": [1, 2, 3],
                },  # Input
                (None, None, None),  # Output
                {
                    "nu": 3,
                    "missing_marker": np.nan,
                    "model2_Cz_Full": False,
                },  # DPADModelArgs
            ],
            [
                {
                    "Y": Y.T,
                    "Z": Z.T,
                    "U": U.T,
                    "allX_steps": [X_step[s].T for s in [1]],
                    "allXp2_steps": [X2_step[s].T for s in [1]],
                    "allZp_steps": [Z2_step[s].T for s in [1]],
                    "steps_ahead": [1],
                },  # Input
                (
                    [
                        X2_step[s].T for s in [1]
                    ],  # Cat of allXp2_steps (allX_steps if model2_Cz_Full) and Y (if has_Dyz) and U (if nu > 0), shifted appropriately
                    [
                        Z2_step[s].T for s in [1]
                    ],  # allZp_steps prior, shifted appropriately
                    [
                        Z.T for _ in [1]
                    ],  # Repetition of Z output with the original timestep
                ),  # Output
                {
                    "nu": 0,
                    "missing_marker": np.nan,
                    "model2_Cz_Full": False,
                    "has_Dyz": False,
                },  # DPADModelArgs
            ],
            [
                {
                    "Y": Y.T,
                    "Z": Z.T,
                    "U": U.T,
                    "allX_steps": [X_step[s].T for s in [1, 1, 1]],
                    "allXp2_steps": [X2_step[s].T for s in [1, 1, 1]],
                    "allZp_steps": [Z2_step[s].T for s in [1, 1, 1]],
                    "steps_ahead": [1, 3, 5],
                },  # Input
                (
                    [
                        X2_step[s].T for s in [1, 3, 5]
                    ],  # Cat of allXp2_steps (allX_steps if model2_Cz_Full) and Y (if has_Dyz) and U (if nu > 0), shifted appropriately
                    [
                        Z2_step[s].T for s in [1, 3, 5]
                    ],  # allZp_steps prior, shifted appropriately
                    [
                        Z.T for _ in [1, 3, 5]
                    ],  # Repetition of Z output with the original timestep
                ),  # Output
                {
                    "nu": 0,
                    "missing_marker": np.nan,
                    "model2_Cz_Full": False,
                    "has_Dyz": False,
                },  # DPADModelArgs
            ],
            [
                {
                    "Y": Y.T,
                    "Z": Z.T,
                    "U": U.T,
                    "allX_steps": [X_step[s].T for s in [1, 1, 1]],
                    "allXp2_steps": [X2_step[s].T for s in [1, 1, 1]],
                    "allZp_steps": [Z2_step[s].T for s in [1, 1, 1]],
                    "steps_ahead": [1, 3, 5],
                },  # Input
                (
                    [
                        X_step[s].T for s in [1, 3, 5]
                    ],  # Cat of allXp2_steps (allX_steps if model2_Cz_Full) and Y (if has_Dyz) and U (if nu > 0), shifted appropriately
                    None,  # allZp_steps prior, shifted appropriately; None if model2_Cz_Full
                    [
                        Z.T for _ in [1, 3, 5]
                    ],  # Repetition of Z output with the original timestep
                ),  # Output
                {
                    "nu": 0,
                    "missing_marker": np.nan,
                    "model2_Cz_Full": True,
                    "has_Dyz": False,
                },  # DPADModelArgs
            ],
        ]
        for caseInd, case in enumerate(testCases):
            input_arg = case[0]
            output_correct = case[1]
            DPADModelArgs = case[2]
            with self.subTest(caseInd=caseInd, input_arg=input_arg):
                s = DPADModel(**DPADModelArgs)
                output = s.prepare_inputs_to_model2_Cz(**input_arg)
                np.testing.assert_equal(output, output_correct)

    def test_DPADModel_prepare_inputs_to_model1(self):
        Y = np.array(
            [
                0.10 + np.arange(1, 21, 1),
                0.11 + np.arange(1, 21, 1),
                0.12 + np.arange(1, 21, 1),
            ]
        ).T
        Z = np.array(
            [
                0.20 + np.arange(1, 21, 1),
                0.21 + np.arange(1, 21, 1),
                0.22 + np.arange(1, 21, 1),
            ]
        ).T
        Z2 = np.array(
            [
                0.30 + np.arange(1, 21, 1),
                0.31 + np.arange(1, 21, 1),
                0.32 + np.arange(1, 21, 1),
            ]
        ).T
        U = np.array(
            [
                0.80 + np.arange(1, 21, 1),
                0.81 + np.arange(1, 21, 1),
                0.82 + np.arange(1, 21, 1),
            ]
        ).T
        X = np.array(
            [
                0.90 + np.arange(1, 21, 1),
                0.91 + np.arange(1, 21, 1),
                0.92 + np.arange(1, 21, 1),
            ]
        ).T
        X2 = np.array(
            [
                0.60 + np.arange(1, 21, 1),
                0.61 + np.arange(1, 21, 1),
                0.62 + np.arange(1, 21, 1),
            ]
        ).T
        YZ = np.concatenate((Y, Z), axis=1)
        YU = np.concatenate((Y, U), axis=1)
        XU = np.concatenate((X, U), axis=1)
        XY = np.concatenate((X, Y), axis=1)
        XYU = np.concatenate((X, Y, U), axis=1)
        Y_step = getCorrectShiftedVersions(Y)
        Z_step = getCorrectShiftedVersions(Z)
        Z2_step = getCorrectShiftedVersions(Z2)
        U_step = getCorrectShiftedVersions(U)
        X_step = getCorrectShiftedVersions(X)
        X2_step = getCorrectShiftedVersions(X2)
        YZ_step = getCorrectShiftedVersions(YZ)
        XU_step = getCorrectShiftedVersions(XU)
        XY_step = getCorrectShiftedVersions(XY)
        XYU_step = getCorrectShiftedVersions(XYU)

        testCases = [
            # Cases without input
            [
                {"Y": None, "U": None},  # Input
                (None, None),  # Output
                {"nu": 3, "has_Dyz": False},  # DPADModelArgs
            ],
            [
                {"Y": Y.T, "U": None},  # Input
                (
                    Y.T,  # Cat of Y and U (if nu > 0)
                    None,  # Cat of Y (if has_Dyz) and U (is not None)
                ),  # Output
                {"nu": 0, "has_Dyz": False},  # DPADModelArgs
            ],
            [
                {"Y": Y.T, "U": U.T},  # Input
                (
                    YU.T,  # Cat of Y and U (if nu > 0)
                    U.T,  # Cat of Y (if has_Dyz) and U (is not None)
                ),  # Output
                {"nu": 3, "has_Dyz": False},  # DPADModelArgs
            ],
            [
                {"Y": Y.T, "U": U.T},  # Input
                (
                    YU.T,  # Cat of Y and U (if nu > 0)
                    YU.T,  # Cat of Y (if has_Dyz) and U (is not None)
                ),  # Output
                {"nu": 3, "has_Dyz": True},  # DPADModelArgs
            ],
        ]
        for caseInd, case in enumerate(testCases):
            input_arg = case[0]
            output_correct = case[1]
            DPADModelArgs = case[2]
            with self.subTest(caseInd=caseInd, input_arg=input_arg):
                s = DPADModel(**DPADModelArgs)
                output = s.prepare_inputs_to_model1(**input_arg)
                np.testing.assert_equal(output, output_correct)

    def test_DPADModel_prepare_inputs_to_model2(self):
        np.random.seed(42)

        sysCode = "nyR3_3_nzR3_3_nuR3_3_NxR6_6_N1R3_3"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        setToLSSMArgsC = {
            "model1_Cy_Full": False,
            "model2_Cz_Full": False,
            "allow_nonzero_Cz2": True,
        }
        sysSettings["predictor_form"] = (
            "allow_nonzero_Cz2" in setToLSSMArgsC
            and not setToLSSMArgsC["allow_nonzero_Cz2"]
        )
        sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)

        N0 = 10

        DPADModelArgs = {}
        s1 = copy.deepcopy(sOrig)
        s1.makeA_KCBlockDiagonal()
        setToLSSMArgsC["ignore_Zero_A_topRight"] = True  #

        s2, E = createDPADModelFromLSSM(s1, DPADModelArgs, setToLSSMArgsC)

        Y = np.array(
            [
                0.10 + np.arange(1, 21, 1),
                0.11 + np.arange(1, 21, 1),
                0.12 + np.arange(1, 21, 1),
            ]
        ).T
        Z = np.array(
            [
                0.20 + np.arange(1, 21, 1),
                0.21 + np.arange(1, 21, 1),
                0.22 + np.arange(1, 21, 1),
            ]
        ).T
        Z2 = np.array(
            [
                0.30 + np.arange(1, 21, 1),
                0.31 + np.arange(1, 21, 1),
                0.32 + np.arange(1, 21, 1),
            ]
        ).T
        U = np.array(
            [
                0.80 + np.arange(1, 21, 1),
                0.81 + np.arange(1, 21, 1),
                0.82 + np.arange(1, 21, 1),
            ]
        ).T
        X = np.array(
            [
                0.90 + np.arange(1, 21, 1),
                0.91 + np.arange(1, 21, 1),
                0.92 + np.arange(1, 21, 1),
            ]
        ).T
        X2 = np.array(
            [
                0.60 + np.arange(1, 21, 1),
                0.61 + np.arange(1, 21, 1),
                0.62 + np.arange(1, 21, 1),
            ]
        ).T
        YZ = np.concatenate((Y, Z), axis=1)
        YU = np.concatenate((Y, U), axis=1)
        XU = np.concatenate((X, U), axis=1)
        XY = np.concatenate((X, Y), axis=1)
        XYU = np.concatenate((X, Y, U), axis=1)
        Y_step = getCorrectShiftedVersions(Y)
        Z_step = getCorrectShiftedVersions(Z)
        Z2_step = getCorrectShiftedVersions(Z2)
        U_step = getCorrectShiftedVersions(U)
        X_step = getCorrectShiftedVersions(X)
        X2_step = getCorrectShiftedVersions(X2)
        YZ_step = getCorrectShiftedVersions(YZ)
        XU_step = getCorrectShiftedVersions(XU)
        XY_step = getCorrectShiftedVersions(XY)
        XYU_step = getCorrectShiftedVersions(XYU)

        testCases = [
            # Cases without input
            [
                {
                    "model1_Cy": None,
                    "Y": None,
                    "U": None,
                    "allXp1_steps": None,
                    "allXp1U_steps": None,
                },  # Input
                (None, None, None),  # Output
                {"nu": 3, "n1": 1},  # DPADModelArgs
            ],
            [
                {
                    "model1_Cy": s2.model1_Cy,
                    "Y": Y.T,
                    "U": U.T,
                    "allXp1_steps": [X_step[s].T for s in [1]],
                    "allXp1U_steps": [XU_step[s].T for s in [1]],
                },  # Input
                (
                    XY.T,  # Cat of allXp1_steps[0] (is n1 > 0) and Y and U (if nu > 0), original time steps (no need for extra shifts)
                    [
                        s2.model1_Cy.predict(XU_step[s].T) for s in [1]
                    ],  # allYp_steps prior, obtained by passing allXp1U_steps to model1_Cy (if available, otherwise, None)
                    [X_step[s].T for s in [1]],  # allXp1_steps if n1>0 else None
                ),  # Output
                {"nu": 0, "n1": 1},  # DPADModelArgs
            ],
            [
                {
                    "model1_Cy": s2.model1_Cy,
                    "Y": Y.T,
                    "U": U.T,
                    "allXp1_steps": [X_step[s].T for s in [1]],
                    "allXp1U_steps": [XU_step[s].T for s in [1]],
                },  # Input
                (
                    XYU.T,  # Cat of allXp1_steps[0] (is n1 > 0) and Y and U (if nu > 0), original time steps (no need for extra shifts)
                    [
                        s2.model1_Cy.predict(XU_step[s].T) for s in [1]
                    ],  # allYp_steps prior, obtained by passing allXp1U_steps to model1_Cy (if available, otherwise, None)
                    [X_step[s].T for s in [1]],  # allXp1_steps if n1>0 else None
                ),  # Output
                {"nu": 3, "n1": 1},  # DPADModelArgs
            ],
            [
                {
                    "model1_Cy": s2.model1_Cy,
                    "Y": Y.T,
                    "U": U.T,
                    "allXp1_steps": [X_step[s].T for s in [1]],
                    "allXp1U_steps": [XU_step[s].T for s in [1]],
                },  # Input
                (
                    YU.T,  # Cat of allXp1_steps[0] (is n1 > 0) and Y and U (if nu > 0), original time steps (no need for extra shifts)
                    None,  # allYp_steps prior, obtained by passing allXp1U_steps to model1_Cy (if available, otherwise, None)
                    None,  # allXp1_steps if n1>0 else None
                ),  # Output
                {"nu": 3, "n1": 0},  # DPADModelArgs
            ],
        ]
        for caseInd, case in enumerate(testCases):
            input_arg = case[0]
            output_correct = case[1]
            DPADModelArgs = case[2]
            with self.subTest(caseInd=caseInd, input_arg=input_arg):
                s = DPADModel(**DPADModelArgs)
                output = s.prepare_inputs_to_model2(**input_arg)
                np.testing.assert_equal(output, output_correct)

    def test_DPADModel_multiStepAhead(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        setToLSSMArgsCases = [
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": True,
            },
            {"model1_Cy_Full": True, "model2_Cz_Full": True, "allow_nonzero_Cz2": True},
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
        ]

        failCount = 0
        failIndsAll = []
        failErrsAll = []

        for setToLSSMArgs in setToLSSMArgsCases:
            with self.subTest(setToLSSMArgs=setToLSSMArgs):

                failInds = []
                failErrs = []
                # numTests = 100
                for ci in range(numTests):
                    setToLSSMArgsC = copy.deepcopy(setToLSSMArgs)
                    sysSettings["predictor_form"] = (
                        "allow_nonzero_Cz2" in setToLSSMArgsC
                        and not setToLSSMArgsC["allow_nonzero_Cz2"]
                    )
                    sysSettings["Dyz"] = ci % 2  # Include Dyz for some models

                    attemptCnt = 0
                    while attemptCnt < 10:
                        try:
                            attemptCnt += 1
                            sOrig, sysU, zErrSys = generateRandomLinearModel(
                                sysSettings
                            )

                            N0 = 10

                            DPADModelArgs = {}
                            x0 = None
                            if (
                                ci % 4 == 0
                            ):  # Make some models trial based with preset initial_state
                                setToLSSMArgsC["stateful"] = False
                                x0 = np.random.randn(sOrig.state_dim)
                                DPADModelArgs["block_samples"] = N0
                            else:
                                x0 = None
                            setToLSSMArgsC["x0"] = x0

                            s = copy.deepcopy(sOrig)
                            # print('Testing system {}/{}'.format(1+ci, numTests))

                            n1 = len(s.zDims)
                            s.makeA_KCBlockDiagonal()
                            setToLSSMArgsC["ignore_Zero_A_topRight"] = True  #
                            if n1 > 0 and n1 < s.state_dim:
                                A12 = s.A[:n1, n1:]
                                s.A[:n1, n1:] = 0
                                setattr(
                                    s, "useA_KC_plus_KC_in_KF", True
                                )  # So that Kalman reconstructs still uses the correct A_KC
                                print(
                                    "WARNING: Current forward prediction only supports models that have blocked A. HERE just to test DPADModel, we will force this to our model and BREAK the internal consistency of noises, Kalman gain, etc"
                                )
                                print(f"Setting A12 from {A12} to {s.A[:n1, n1:]}")

                            s2, E = createDPADModelFromLSSM(
                                s, DPADModelArgs, setToLSSMArgsC
                            )
                            # s.applySimTransform(E) # IMPORTANT: To make the basis the same as the one the RNN model has used
                            break
                        except Exception as e:
                            print(
                                f"Attempt {attemptCnt} in creating a random model failed: {e}"
                            )

                    num_trials = np.random.randint(1, 4)
                    N = N0 * num_trials
                    if s.input_dim:
                        U, XU = sysU.generateRealization(N)
                    else:
                        U = None
                    Y, X = s.generateRealizationWithKF(N, u=U)

                    allZp2, allYp2, allXp2 = s2.predict(Y, U=U)

                    # Now using the mutli step ahead code
                    s2.set_steps_ahead([1])
                    allZp2b, allYp2b, allXp2b = s2.predict(Y, U=U)
                    try:
                        np.testing.assert_allclose(
                            allXp2, allXp2b, rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            allYp2, allYp2b, rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            allZp2, allZp2b, rtol=1e-3, atol=1e-6
                        )
                    except Exception as e:
                        failInds.append(ci)
                        failErrs.append(e)
                        break

                    # The order of steps ahead should not matter
                    setToLSSMArgsC["steps_ahead"] = [2, 5]
                    s3, E3 = createDPADModelFromLSSM(s, DPADModelArgs, setToLSSMArgsC)

                    # s.applySimTransform(E3) # IMPORTANT: To make the basis the same as the one the RNN model has used

                    try:
                        outs1 = s3.predict(Y, U=U)

                        # The order of steps ahead should not matter
                        s3.set_steps_ahead([5, 2])
                        outs2 = s3.predict(Y, U=U)

                        np.testing.assert_allclose(
                            outs1[0], outs2[1], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[1], outs2[0], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[2], outs2[3], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[3], outs2[2], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[4], outs2[5], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[5], outs2[4], rtol=1e-3, atol=1e-6
                        )
                    except Exception as e:
                        failInds.append(ci)
                        failErrs.append(e)
                        break

                failCount += len(failInds)
                failIndsAll.append(failInds)
                failErrsAll.append(failErrs)
                if len(failInds) > 0:
                    print(
                        "{} (with {}) => {}/{} random systems (indices: {}) failed: \n{}".format(
                            self.id(),
                            setToLSSMArgs,
                            len(failInds),
                            numTests,
                            failInds,
                            failErrs,
                        )
                    )
                else:
                    print(
                        "{} (with {}) => Ok: Tested with {} random systems, all were ok!".format(
                            self.id(), setToLSSMArgs, numTests
                        )
                    )

    def test_DPADModel_multiStepAhead_v2(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_0_NxR2_5_N1R0_3"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        setToLSSMArgsCases = [
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": True,
            },
            {"model1_Cy_Full": True, "model2_Cz_Full": True, "allow_nonzero_Cz2": True},
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
        ]

        failCount = 0
        failIndsAll = []
        failErrsAll = []

        for setToLSSMArgsInd, setToLSSMArgs in enumerate(setToLSSMArgsCases):
            with self.subTest(setToLSSMArgs=setToLSSMArgs):

                failInds = []
                failErrs = []
                # numTests = 100
                for ci in range(numTests):
                    setToLSSMArgsC = copy.deepcopy(setToLSSMArgs)
                    sysSettings["predictor_form"] = (
                        "allow_nonzero_Cz2" in setToLSSMArgsC
                        and not setToLSSMArgsC["allow_nonzero_Cz2"]
                    )
                    # sysSettings['Dyz'] = ci % 2 # Include Dyz for some models
                    sysSettings["Dyz"] = 0  # Include Dyz for some models

                    attemptCnt = 0
                    while attemptCnt < 10:
                        try:
                            attemptCnt += 1
                            sOrig, sysU, zErrSys = generateRandomLinearModel(
                                sysSettings
                            )

                            N0 = 10

                            DPADModelArgs = {}
                            x0 = None
                            if (
                                ci % 4 == 0
                            ):  # Make some models trial based with preset initial_state
                                setToLSSMArgsC["stateful"] = False
                                x0 = np.random.randn(sOrig.state_dim)
                                DPADModelArgs["block_samples"] = N0
                            else:
                                x0 = None
                            setToLSSMArgsC["x0"] = x0

                            s = copy.deepcopy(sOrig)
                            # print('Testing system {}/{}'.format(1+ci, numTests))

                            n1 = len(s.zDims)
                            s.makeA_KCBlockDiagonal()
                            setToLSSMArgsC["ignore_Zero_A_topRight"] = True  #
                            # if n1 > 0 and n1 < s.state_dim: # NOT NEEDED FOR THIS METHOD OF FORWARD PREDICTION
                            #     A12 = np.array(s.A[:n1, n1:])
                            #     s.A[:n1, n1:] = 0
                            #     setattr(s, 'useA_KC_plus_KC_in_KF', True) # So that Kalman reconstructs still uses the correct A_KC
                            #     print('WARNING: Current forward prediction only supports models that have blocked A. HERE just to test DPADModel, we will force this to our model and BREAK the internal consistency of noises, Kalman gain, etc')
                            #     print(f'Setting A12 from {A12} to {s.A[:n1, n1:]}')

                            s2, E = createDPADModelFromLSSM(
                                s, DPADModelArgs, setToLSSMArgsC
                            )
                            # s.applySimTransform(E) # IMPORTANT: To make the basis the same as the one the RNN model has used
                            break
                        except Exception as e:
                            print(
                                f"Attempt {attemptCnt} in creating a random model failed: {e}"
                            )

                    sBack = s2.getLSSM()
                    if np.linalg.norm(E - np.eye(E.shape[0])) > 1e-12:
                        s.applySimTransform(
                            E
                        )  # IMPORTANT: To make the basis the same as the one the RNN model has used

                    num_trials = np.random.randint(1, 4)
                    N = N0 * num_trials
                    if s.input_dim:
                        U, XU = sysU.generateRealization(N)
                    else:
                        U = None
                    Y, X = s.generateRealizationWithKF(N, u=U)

                    YTrials = [
                        Y[(ti * N0) : ((ti + 1) * N0)] for ti in range(int(N / N0))
                    ]
                    UTrials = (
                        [U[(ti * N0) : ((ti + 1) * N0)] for ti in range(int(N / N0))]
                        if U is not None
                        else None
                    )

                    allZp0Trials, allYp0Trials, allXp0Trials = s.predict(
                        YTrials, U=UTrials, x0=x0
                    )
                    allZp0 = np.concatenate(allZp0Trials)
                    allYp0 = np.concatenate(allYp0Trials)
                    allXp0 = np.concatenate(allXp0Trials)

                    allZp2, allYp2, allXp2 = s2.predict(Y, U=U, x0=x0)

                    # Now using the mutli step ahead code
                    s2.set_steps_ahead([1])
                    allZp2b, allYp2b, allXp2b = s2.predict(Y, U=U, x0=x0)
                    try:
                        np.testing.assert_allclose(
                            allXp2, allXp2b, rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            allYp2, allYp2b, rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            allZp2, allZp2b, rtol=1e-3, atol=1e-6
                        )
                    except Exception as e:
                        failInds.append(ci)
                        failErrs.append(e)
                        break

                    # The order of steps ahead should not matter
                    setToLSSMArgsC["steps_ahead"] = [2, 5]
                    s3, E3 = createDPADModelFromLSSM(s, DPADModelArgs, setToLSSMArgsC)

                    # Second method with passing generated outputs as observations
                    s.steps_ahead = [1, 2, 5]
                    s3.set_steps_ahead([1, 2, 5])
                    s3.set_multi_step_with_data_gen(True)

                    # s4, E3 = createDPADModelFromLSSM(s, DPADModelArgs, setToLSSMArgsC)
                    # s4.set_steps_ahead([1,2,5])
                    # s4.set_multi_step_with_data_gen(True, noise_samples=100)

                    try:
                        if x0 is not None:  # Trial based
                            outs0 = s.predict(YTrials, U=UTrials, x0=x0)
                            outs0 = [np.concatenate(out) for out in outs0]
                        else:
                            outs0 = s.predict(Y, U=U, x0=x0)

                        outs1 = s3.predict(Y, U=U, x0=x0)
                        for o0, o1 in zip(outs0, outs1):
                            np.testing.assert_allclose(o0, o1, rtol=1e-3, atol=1e-5)

                        # Now with noise averaging
                        # outs4 = s4.predict(Y, U=U, x0=x0)
                        # for o0, o1 in zip(outs0, outs4):
                        #     np.testing.assert_allclose(o0, o1, rtol=1e-3, atol=1e-5)

                        # The order of steps ahead should not matter
                        s3.set_steps_ahead([5, 2], update_rnn_model_steps=False)
                        outs2 = s3.predict(Y, U=U, x0=x0)

                        np.testing.assert_allclose(
                            outs1[1], outs2[1], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[2], outs2[0], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[4], outs2[3], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[5], outs2[2], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[7], outs2[5], rtol=1e-3, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            outs1[8], outs2[4], rtol=1e-3, atol=1e-6
                        )

                    except Exception as e:
                        failInds.append(ci)
                        failErrs.append(e)
                        print(e)
                        break

                failCount += len(failInds)
                failIndsAll.append(failInds)
                failErrsAll.append(failErrs)
                if len(failInds) > 0:
                    print(
                        "{} (with {}) => {}/{} random systems (indices: {}) failed: \n{}".format(
                            self.id(),
                            setToLSSMArgs,
                            len(failInds),
                            numTests,
                            failInds,
                            failErrs,
                        )
                    )
                else:
                    print(
                        "{} (with {}) => Ok: Tested with {} random systems, all were ok!".format(
                            self.id(), setToLSSMArgs, numTests
                        )
                    )

        if failCount > 0:
            raise (
                Exception(
                    f"{self.id()} - There were some errors. \nFailure indices: {failIndsAll}\nFailure errors: {failErrsAll}"
                )
            )

    def test_DPADModel_setAsLSSM(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        setToLSSMArgsCases = [
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": True,
            },
            {"model1_Cy_Full": True, "model2_Cz_Full": True, "allow_nonzero_Cz2": True},
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
        ]

        failCount = 0
        failStepAheadsAll = []
        failIndsAll = []
        failErrsAll = []
        failModelsAll = []

        for setToLSSMArgs in setToLSSMArgsCases:
            with self.subTest(setToLSSMArgs=setToLSSMArgs):

                failStepAheads = []
                failInds = []
                failErrs = []
                failModels = []
                # numTests = 100
                for ci in range(numTests):
                    setToLSSMArgsC = copy.deepcopy(setToLSSMArgs)
                    sysSettings["predictor_form"] = (
                        "allow_nonzero_Cz2" in setToLSSMArgsC
                        and not setToLSSMArgsC["allow_nonzero_Cz2"]
                    )
                    sysSettings["Dyz"] = ci % 2  # Include Dyz for some models
                    attemptCnt = 0
                    while attemptCnt < 10:
                        try:
                            attemptCnt += 1
                            sOrig, sysU, zErrSys = generateRandomLinearModel(
                                sysSettings
                            )

                            N0 = 10

                            DPADModelArgs = {}
                            x0 = None
                            if (
                                ci % 4 == 0
                            ):  # Make some models trial based with preset initial_state
                                setToLSSMArgsC["stateful"] = False
                                x0 = np.random.randn(sOrig.state_dim)
                                DPADModelArgs["block_samples"] = N0
                            else:
                                x0 = None
                            setToLSSMArgsC["x0"] = x0

                            s = copy.deepcopy(sOrig)
                            # print('Testing system {}/{}'.format(1+ci, numTests))
                            # s.makeSZero()
                            # s.makeA_and_A_KCBlockDiagonal()

                            n1 = len(s.zDims)
                            s.makeA_KCBlockDiagonal()
                            setToLSSMArgsC["ignore_Zero_A_topRight"] = True  #
                            if n1 > 0 and n1 < s.state_dim:
                                A12 = s.A[:n1, n1:]
                                s.A[:n1, n1:] = 0
                                setattr(
                                    s, "useA_KC_plus_KC_in_KF", True
                                )  # So that Kalman reconstructs still uses the correct A_KC
                                print(
                                    "WARNING: Current forward prediction only supports models that have blocked A. HERE just to test DPADModel, we will force this to our model and BREAK the internal consistency of noises, Kalman gain, etc"
                                )
                                print(f"Setting A12 from {A12} to {s.A[:n1, n1:]}")

                            s2, E = createDPADModelFromLSSM(
                                s, DPADModelArgs, setToLSSMArgsC
                            )
                            break
                        except Exception as e:
                            print(
                                f"Attempt {attemptCnt} in creating a random model failed: {e}"
                            )

                    sBack = s2.getLSSM()
                    if np.linalg.norm(E - np.eye(E.shape[0])) > 1e-12:
                        s.applySimTransform(
                            E
                        )  # IMPORTANT: To make the basis the same as the one the RNN model has used

                    num_trials = 1  # np.random.randint(1,4)
                    N = N0 * num_trials
                    if s.input_dim:
                        U, XU = sysU.generateRealization(N)
                    else:
                        U = None
                    Y, X = s.generateRealizationWithKF(N, u=U)

                    allZp_1step_cat, allYp_1step_cat, allXp_1step_cat = s.predict(
                        Y, U=U, x0=x0
                    )
                    allZp2_1step_cat, allYp2_1step_cat, allXp2_1step_cat = s2.predict(
                        Y, U=U, x0=x0
                    )

                    steps_ahead = [1, 2, 5]
                    for step_ahead in steps_ahead:
                        setattr(s, "steps_ahead", [step_ahead])
                        s2.set_steps_ahead([step_ahead])
                        setToLSSMArgsC["steps_ahead"] = [step_ahead]
                        E = s2.setToLSSM(
                            s, **setToLSSMArgsC
                        )  # Just to change steps ahead by rebuilding the model (in case the network haschanges)
                        if np.linalg.norm(E - np.eye(E.shape[0])) > 1e-12:
                            raise (
                                Exception(
                                    "s should not need any additional similarity transforms before being settable as DPADModel"
                                )
                            )

                        if (
                            num_trials > 1 and x0 is not None
                        ):  # When we have multiple trials, we expect the x0 to reset in the beginning of each
                            for ti in range(num_trials):
                                allZpThis, allYpThis, allXpThis = s.predict(
                                    Y=Y[(ti * N0) : ((ti + 1) * N0), :],
                                    U=(
                                        U[(ti * N0) : ((ti + 1) * N0), :]
                                        if U is not None
                                        else None
                                    ),
                                    x0=x0,
                                )
                                if ti == 0:
                                    allZp = allZpThis
                                    allYp = allYpThis
                                    allXp = allXpThis
                                else:
                                    allZp = np.concatenate((allZp, allZpThis), axis=0)
                                    allYp = np.concatenate((allYp, allYpThis), axis=0)
                                    allXp = np.concatenate((allXp, allXpThis), axis=0)
                        else:
                            allZp, allYp, allXp = s.predict(Y, U=U, x0=x0)

                        try:
                            allZp2, allYp2, allXp2 = s2.predict(Y, U=U, x0=x0)

                            np.testing.assert_allclose(
                                allXp, allXp2, rtol=1e-3, atol=1e-6
                            )
                            np.testing.assert_allclose(
                                allYp, allYp2, rtol=1e-3, atol=1e-6
                            )
                            np.testing.assert_allclose(
                                allZp, allZp2, rtol=1e-3, atol=1e-6
                            )
                        except Exception as e:
                            failStepAheads.append(step_ahead)
                            failInds.append(ci)
                            failErrs.append(e)
                            failModels.append(sOrig)

                failCount += len(failInds)
                failStepAheadsAll.append(failStepAheads)
                failIndsAll.append(failInds)
                failErrsAll.append(failErrs)
                failModels.append(failModels)
                if len(failInds) > 0:
                    print(
                        "{} (with {}) => {}/{} random systems (indices: {}) failed: \n{}".format(
                            self.id(),
                            setToLSSMArgs,
                            len(failInds),
                            numTests,
                            failInds,
                            failErrs,
                        )
                    )
                else:
                    print(
                        "{} (with {}) => Ok: Tested with {} random systems, all were ok!".format(
                            self.id(), setToLSSMArgs, numTests
                        )
                    )

        if failCount > 0:
            raise (
                Exception(
                    f"{self.id()} - There were some errors. \nFailure indices: {failIndsAll}\nFailure errors: {failErrsAll}"
                )
            )

    def test_DPADModel_getLSSM(self):
        np.random.seed(42)

        setToLSSMArgsCases = [
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": True,
            },
            {"model1_Cy_Full": True, "model2_Cz_Full": True, "allow_nonzero_Cz2": True},
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
        ]

        for setToLSSMArgs in setToLSSMArgsCases:
            with self.subTest(setToLSSMArgs=setToLSSMArgs):

                sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
                sysSettings = getSysSettingsFromSysCode(sysCode)

                # numTests = 100
                for ci in range(numTests):
                    sysSettings["predictor_form"] = (
                        "allow_nonzero_Cz2" in setToLSSMArgs
                        and not setToLSSMArgs["allow_nonzero_Cz2"]
                    )
                    sysSettings["Dyz"] = ci % 2  # Include Dyz for some models
                    sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                    s = copy.deepcopy(sOrig)
                    # print('Testing system {}/{}'.format(1+ci, numTests))

                    setToLSSMArgs["has_Dyz"] = sysSettings["Dyz"]
                    setToLSSMArgs["remove_flat_dims"] = (
                        False  # To avoid dimension mismatch cased by passing dummy potentially flat data to fit
                    )
                    try:
                        s2 = DPADModel()
                        # E = s2.setToLSSM(s, **setToLSSMArgs)
                        Y = np.random.randn(s.output_dim, 1 + 256)
                        Z = np.random.randn(s.Cz.shape[0], 1 + 256)
                        U = np.random.randn(s.input_dim, 1 + 256)
                        s2.fit(
                            Y,
                            Z=Z,
                            U=U,
                            nx=s.state_dim,
                            n1=s.zDims.size,
                            init_model=s,
                            zscore_inputs=False,
                            epochs=0,
                            skip_predictions=True,
                            verbose=False,
                            **setToLSSMArgs,
                        )
                        E = s2.initE
                        accept_zDims = False
                    except Exception as e:
                        print(e)
                        s2 = DPADModel(n1=s.zDims.size + 1, nx=s.state_dim)
                        # E = s2.setToLSSM(s, **setToLSSMArgs)
                        s2.fit(
                            Y,
                            Z=Z,
                            U=U,
                            nx=s.state_dim,
                            n1=s.zDims.size + 1,
                            init_model=s,
                            zscore_inputs=False,
                            epochs=0,
                            skip_predictions=True,
                            verbose=False,
                            **setToLSSMArgs,
                        )
                        E = s2.initE
                        accept_zDims = True

                    EInv = np.linalg.inv(E)
                    sBack = s2.getLSSM()
                    sBack.applySimTransform(EInv)

                    (
                        skipParams,
                        impossibleParams,
                        okParams,
                        errorParams,
                        errorParamsErr,
                    ) = assert_params_are_close(s, sBack)

                    if len(errorParams):
                        errMsg = "FAILED for random sys {}/{} => The following parameter(s) had errors:\n".format(
                            1 + ci, numTests
                        )
                        for p, e in zip(errorParams, errorParamsErr):
                            errMsg += '- Parameter "{}" => Error: "{}"\n'.format(p, e)
                        if accept_zDims and errorParams == ["zDims"]:
                            continue  # Let's not stop the whole test for this impossible diagonalization problem
                        raise (Exception(errMsg))

                if len(okParams):
                    print("Parameter(s) {} were Ok!".format(", ".join(okParams)))

                if len(skipParams):
                    print(
                        "Skipped parameter(s) {} which was not available in the recovered model".format(
                            ", ".join(skipParams)
                        )
                    )

                if len(impossibleParams):
                    print(
                        'Skipped parameter(s) "{}" which are no longer recoverable after the roundtrip to predictor form'.format(
                            ", ".join(impossibleParams)
                        )
                    )

                print(
                    "{}({}) => Ok: Tested with {} random systems".format(
                        self.id(), setToLSSMArgs, numTests
                    )
                )

    def test_DPADModel_discardModel(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        setToLSSMArgsCases = [
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": True,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": True,
            },
            {"model1_Cy_Full": True, "model2_Cz_Full": True, "allow_nonzero_Cz2": True},
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": False,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": False,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
            {
                "model1_Cy_Full": True,
                "model2_Cz_Full": True,
                "allow_nonzero_Cz2": False,
            },
        ]

        failCount = 0
        failIndsAll = []
        failErrsAll = []

        for setToLSSMArgs in setToLSSMArgsCases:
            with self.subTest(setToLSSMArgs=setToLSSMArgs):

                failInds = []
                failErrs = []
                # numTests = 100
                for ci in range(numTests):
                    setToLSSMArgsC = copy.deepcopy(setToLSSMArgs)

                    sysSettings["predictor_form"] = (
                        "allow_nonzero_Cz2" in setToLSSMArgs
                        and not setToLSSMArgs["allow_nonzero_Cz2"]
                    )
                    sysSettings["Dyz"] = ci % 2  # Include Dyz for some models
                    sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                    # sOrig.Dz = 0 * sOrig.Dz
                    s = copy.deepcopy(sOrig)
                    # print('Testing system {}/{}'.format(1+ci, numTests))

                    N0 = 16

                    DPADModelArgs = {}

                    x0 = None
                    if (
                        ci % 4 == 0
                    ):  # Make some models trial based with preset initial_state
                        setToLSSMArgsC["stateful"] = False
                        x0 = np.random.randn(sOrig.state_dim)
                        x0 = x0[:, np.newaxis]
                        DPADModelArgs["block_samples"] = N0
                        DPADModelArgs["batch_size"] = 8
                    else:
                        x0 = None
                    setToLSSMArgsC["x0"] = x0
                    setToLSSMArgsC["ignore_Zero_A_KC_topRight"] = True
                    # setToLSSMArgsC['Cz_args'] = {'units': [], 'use_bias': False, 'activation': 'linear', 'output_activation': 'exponential'}

                    min_trials = 64 if ci % 2 == 0 else 16
                    num_trials = min_trials + np.random.randint(1, 4)
                    N = N0 * num_trials

                    if s.input_dim:
                        U, XU = sysU.generateRealization(N)
                        UT = U.T
                    else:
                        U, UT = None, None
                    Y, X = s.generateRealizationWithKF(N, u=U)
                    Z = np.sin(X)

                    s2 = DPADModel(**DPADModelArgs)
                    args = DPADModel.prepare_args("DPAD_disU_Cz1HL64U_n1_256")
                    if ci % 3 == 0:  # Also provide validation data
                        args.update(
                            {
                                "Y_validation": Y.T,
                                "Z_validation": Z.T,
                                "U_validation": UT,
                            }
                        )
                    elif ci % 2 == 0:
                        args.update({"create_val_from_training": True})
                    if ci % 5 == 0:  # Also provide validation data
                        steps_ahead = [1, 3, 5]
                        args.update({"steps_ahead": steps_ahead})
                    else:
                        steps_ahead = 1
                    try:
                        s2.fit(
                            Y.T,
                            Z=Z.T,
                            U=UT,
                            nx=s.state_dim,
                            n1=s.state_dim,
                            epochs=2,
                            **args,
                        )

                        preds1 = s2.predict(Y, U=U, x0=x0)

                        # Make sure model stays the same after discardModels and restoreModels
                        s2.discardModels()
                        # # Save to file
                        # tmp_file = './temptest.p'
                        # if tmp_file is not None: # Also try saving to and loading form file
                        #     pickle_save(tmp_file, {'s2': s2})
                        #     s2 = pickle_load(tmp_file)['s2']
                        #     os.remove(tmp_file)
                        s2.restoreModels()
                        preds2 = s2.predict(Y, U=U, x0=x0)

                        for pred1, pred2 in zip(preds1, preds2):
                            np.testing.assert_allclose(pred1, pred2, rtol=1e-3)
                    except Exception as e:
                        failInds.append(ci)
                        failErrs.append(e)

                failCount += len(failInds)
                failIndsAll.append(failInds)
                failErrsAll.append(failErrs)
                if len(failInds) > 0:
                    print(
                        "{} (with {}) => {}/{} random systems (indices: {}) failed: \n{}".format(
                            self.id(),
                            setToLSSMArgs,
                            len(failInds),
                            numTests,
                            failInds,
                            failErrs,
                        )
                    )
                else:
                    print(
                        "{} (with {}) => Ok: Tested with {} random systems, all were ok!".format(
                            self.id(), setToLSSMArgs, numTests
                        )
                    )

        if failCount > 0:
            raise (
                Exception(
                    f"{self.id()} - There were some errors. \nFailure indices: {failIndsAll}\nFailure errors: {failErrsAll}"
                )
            )

    def test_DPADModel_generateRealization(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"

        sysSettings = getSysSettingsFromSysCode(sysCode)

        setToLSSMArgs = {
            "model1_Cy_Full": False,
            "model2_Cz_Full": False,
            "allow_nonzero_Cz2": True,
        }

        failInds = []
        failErrs = []
        # numTests = 100
        for ci in range(numTests):
            sysSettings["predictor_form"] = (
                "allow_nonzero_Cz2" in setToLSSMArgs
                and not setToLSSMArgs["allow_nonzero_Cz2"]
            )
            sysSettings["Dyz"] = ci % 2  # Include Dyz for some models
            sysSettings["Dyz"] = False
            sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
            s = copy.deepcopy(sOrig)
            # print('Testing system {}/{}'.format(1+ci, numTests))

            DPADModelArgs = {}

            N0 = 16
            num_trials = 1 + np.random.randint(1, 4)
            N = N0 * num_trials

            if s.input_dim:
                U, XU = sysU.generateRealization(N)
                UT = U.T
            else:
                U, UT = None, None
            Y, X = s.generateRealizationWithKF(N, u=U)
            Z = np.sin(X)

            setToLSSMArgs["ignore_Zero_A_topRight"] = True  #

            s2, E = createDPADModelFromLSSM(s, DPADModelArgs, setToLSSMArgs)

            eY, eYShaping = genRandomGaussianNoise(N, s.innovCov)
            Y, X = s.generateRealizationWithKF(N, u=U, e=eY)
            Z = (s.Cz @ X.T).T
            if s.input_dim > 0 and hasattr(s, "Dz") and s.Dz is not None:
                Z += (s.Dz @ U.T).T
            Z2, Y2, X2 = s2.generateRealization(N, u=U, eY=eY)

            try:
                np.testing.assert_allclose(X, X2, rtol=1e-3)
                np.testing.assert_allclose(Y, Y2, rtol=1e-2)
                np.testing.assert_allclose(Z, Z2, rtol=1e-2)
            except Exception as e:
                failInds.append(ci)
                failErrs.append(e)

        if len(failInds) > 0:
            print(
                "{} (with {}) => {}/{} random systems (indices: {}) failed: \n{}".format(
                    self.id(),
                    setToLSSMArgs,
                    len(failInds),
                    numTests,
                    failInds,
                    failErrs,
                )
            )
        else:
            print(
                "{} (with {}) => Ok: Tested with {} random systems, all were ok!".format(
                    self.id(), setToLSSMArgs, numTests
                )
            )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
