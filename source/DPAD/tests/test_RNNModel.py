""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Tests RNNModel"""

# pylint: disable=C0103, C0111

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from DPAD.DPADModel import shift_ms_to_1s_series
from DPAD.RNNModel import RNNModel
from DPAD.sim import (
    generateRandomLinearModel,
    genRandomGaussianNoise,
    getSysSettingsFromSysCode,
)
from DPAD.tools.tf_tools import set_global_tf_eagerly_flag

numTests = 10


class TestRNNModel(unittest.TestCase):
    def testRNNModel_initLSSM(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        failInds = []
        failErrs = []

        # numTests = 100
        for ci in range(numTests):
            sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
            s = copy.deepcopy(sOrig)
            # print('Testing system {}/{}'.format(1+ci, numTests))

            block_samples = 64
            stateful = True
            # Tests ARE EXPECTED to FAIL if stateful = False, because at edges of block_samples, initial state of RNN resets, but that does not happen for Kalman.
            # This is why is is important to have stateful=True

            N = (
                20 if ci % 5 > 0 else 200
            )  # Important to also have some longer data to test cass with multiple batches
            if not stateful:
                N = (
                    1 * block_samples if ci % 5 > 0 else 4 * block_samples
                )  # For non-stateful (trial-based RNNs), input data must be a multiple of block_samples
            if s.input_dim:
                U, XU = sysU.generateRealization(N)
                UT = U.T
            else:
                U, UT = None, None
            Y, X = s.generateRealizationWithKF(N, u=U)
            if s.input_dim > 0:
                YU = np.concatenate((Y, U), axis=1)
                UT, YUT = U.T, YU.T
            else:
                YU, YUT = Y, Y.T

            allZp, allYp, allXp = s.predict(Y, U=U)

            model2 = RNNModel(
                initLSSM=s, block_samples=64, batch_size=10, stateful=stateful
            )
            allXp2, allYp2 = model2.predict(YUT, FT_in=UT, use_quick_method=True)[:2]
            outs = model2.predict_with_keras(
                (YUT, UT)
            )  # Just making sure that this can run too

            try:
                np.testing.assert_allclose(allXp, allXp2.T, rtol=1e-3, atol=1e-6)
                np.testing.assert_allclose(allYp, allYp2.T, rtol=1e-3, atol=1e-6)
            except Exception as e:
                failInds.append(ci)
                failErrs.append(e)

        if len(failInds) > 0:
            raise (
                Exception(
                    "{} => {}/{} random systems (indices: {}) failed: \n{}".format(
                        self.id(), len(failInds), numTests, failInds, failErrs
                    )
                )
            )
        else:
            print(
                "{} => Ok: Tested with {} random systems, all were ok!".format(
                    self.id(), numTests
                )
            )

    def testRNNModel_MutliStepAheadPrediction(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        failInds = []
        failStepAheads = []
        failErrs = []

        # numTests = 100
        for ci in range(numTests):
            sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
            s = copy.deepcopy(sOrig)
            # print('Testing system {}/{}'.format(1+ci, numTests))

            N = (
                20 if ci % 5 > 0 else 200
            )  # Important to also have some longer data to test cases with multiple batches
            if s.input_dim:
                U, XU = sysU.generateRealization(N)
                UT = U.T
            else:
                U, UT = None, None
            Y, X = s.generateRealizationWithKF(N, u=U)
            if s.input_dim > 0:
                YU = np.concatenate((Y, U), axis=1)
                UT, YUT = U.T, YU.T
            else:
                YU, YUT = Y, Y.T

            allZp, allYp, allXp = s.predict(Y, U=U)

            model1 = RNNModel(
                initLSSM=s, block_samples=64, batch_size=10, steps_ahead=None
            )
            outs1 = model1.predict(YUT, FT_in=UT, use_quick_method=True)

            steps_ahead = [1, 2, 5, 10]

            model2 = RNNModel(
                initLSSM=s,
                block_samples=64,
                batch_size=10,
                steps_ahead=steps_ahead,
                enable_forward_pred=True,
                multi_step_with_A_KC=True,
            )
            outs2 = model2.predict(YUT, FT_in=UT, use_quick_method=True)

            model3 = RNNModel(
                initLSSM=s,
                block_samples=64,
                batch_size=10,
                steps_ahead=steps_ahead,
                enable_forward_pred=True,
            )
            outs3 = model3.predict(YUT, FT_in=UT, use_quick_method=True)

            model3.set_use_feedthrough_in_fw(True)
            outs3_use_FT_in_fw = model3.predict(YUT, FT_in=UT, use_quick_method=True)

            model4 = RNNModel(
                initLSSM=s,
                block_samples=64,
                batch_size=10,
                steps_ahead=steps_ahead,
                enable_forward_pred=True,
            )
            model4.set_steps_ahead([1])
            outs4 = model4.predict(YUT, FT_in=UT, use_quick_method=True)

            outs_tmp = model2.predict_with_keras(
                (YUT, UT)
            )  # Just making sure that this can run too

            np.testing.assert_allclose(outs1[0], outs2[0], rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(
                outs1[1], outs2[len(steps_ahead)], rtol=1e-3, atol=1e-6
            )
            np.testing.assert_allclose(outs1[0], outs3[0], rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(
                outs1[1], outs3[len(steps_ahead)], rtol=1e-3, atol=1e-6
            )
            np.testing.assert_allclose(outs1[0], outs4[0], rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(outs1[1], outs4[1], rtol=1e-3, atol=1e-6)

            outs3Ys_shifted = shift_ms_to_1s_series(
                outs3[len(steps_ahead) : 2 * len(steps_ahead)],
                steps_ahead,
                time_first=False,
            )

            for saInd, step_ahead in enumerate(steps_ahead):
                thisXp_A_KC = np.array(allXp)
                thisXp = np.array(allXp)
                for stepInd in range(step_ahead - 1):
                    thisXp_A_KC = (s.A_KC @ thisXp_A_KC.T).T
                    thisXp = (s.A @ thisXp.T).T
                thisYp_A_KC = (s.C @ thisXp_A_KC.T).T
                thisYp = (s.C @ thisXp.T).T
                if step_ahead == 1 and s.input_dim > 0:
                    thisYp_A_KC += (s.D @ UT).T
                    thisYp += (s.D @ UT).T

                UTShift = np.concatenate(
                    (
                        UT[:, (step_ahead - 1) :],
                        np.zeros_like(UT[:, : (step_ahead - 1)]),
                    ),
                    axis=1,
                )
                thisYp_use_FT_in_fw = (s.C @ thisXp.T).T + (
                    s.D @ UTShift
                ).T  # Always, even for forecasting

                try:
                    if step_ahead == 1:
                        np.testing.assert_allclose(
                            thisYp_use_FT_in_fw, thisYp, rtol=1e-3, atol=1e-6
                        )
                    np.testing.assert_allclose(
                        thisXp_A_KC, outs2[saInd].T, rtol=1e-3, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        thisYp_A_KC,
                        outs2[saInd + len(steps_ahead)].T,
                        rtol=1e-3,
                        atol=1e-5,
                    )
                    np.testing.assert_allclose(
                        thisXp, outs3[saInd].T, rtol=1e-3, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        thisYp, outs3[saInd + len(steps_ahead)].T, rtol=1e-3, atol=1e-5
                    )
                    np.testing.assert_allclose(
                        thisYp_use_FT_in_fw,
                        outs3_use_FT_in_fw[saInd + len(steps_ahead)].T,
                        rtol=1e-3,
                        atol=1e-5,
                    )
                except Exception as e:
                    failInds.append(ci)
                    failStepAheads.append(step_ahead)
                    failErrs.append(e)

                # Test shift_ms_to_1s_series
                if step_ahead == 1:
                    thisYp_same_time_as_Y = thisYp
                else:
                    thisYp_same_time_as_Y = np.nan * np.ones_like(thisYp)
                    thisYp_same_time_as_Y[(step_ahead - 1) :, :] = thisYp[
                        : (-step_ahead + 1), :
                    ]

                np.testing.assert_allclose(
                    thisYp_same_time_as_Y,
                    outs3Ys_shifted[saInd].T,
                    rtol=1e-3,
                    atol=1e-5,
                )

            pass

        if len(failInds) > 0:
            raise (
                Exception(
                    "{} => {}/{} random systems (indices: {}) failed: \n{}".format(
                        self.id(), len(failInds), numTests, failInds, failErrs
                    )
                )
            )
        else:
            print(
                "{} => Ok: Tested with {} random systems, all were ok!".format(
                    self.id(), numTests
                )
            )

    def testRNNModel_Bidirectional(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_0_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        failInds = []
        failErrs = []

        # numTests = 100
        for ci in range(numTests):
            sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)

            if ci == 0:  # A special simple system that is easy to track
                sOrig, sysU, zErrSys = generateRandomLinearModel(
                    getSysSettingsFromSysCode("nyR1_1_nzR1_1_nuR0_0_NxR1_1_N1R1_1")
                )
                sOrig.changeParams({"A": sOrig.A / sOrig.A * 0.999})
                # Temp, setting up an inconsistent model
                sOrig.A_KC = sOrig.A_KC / sOrig.A_KC * 0.01
                sOrig.K = sOrig.K / sOrig.K * 1
                sOrig.C = sOrig.C / sOrig.C * 1
                sOrig.useA_KC_plus_KC_in_KF = True

            s = copy.deepcopy(sOrig)
            # print('Testing system {}/{}'.format(1+ci, numTests))

            block_samples = 64
            stateful = True
            # Tests ARE EXPECTED to FAIL if stateful = False, because at edges of block_samples, initial state of RNN resets, but that does not happen for Kalman.
            # This is why is is important to have stateful=True

            N = (
                20 if ci % 5 > 0 else 200
            )  # Important to also have some longer data to test cass with multiple batches
            if ci == 0:
                N = 20
            if not stateful:
                N = (
                    1 * block_samples if ci % 5 > 0 else 4 * block_samples
                )  # For non-stateful (trial-based RNNs), input data must be a multiple of block_samples
            if s.input_dim:
                U, XU = sysU.generateRealization(N)
                UT = U.T
            else:
                U, UT = None, None
            Y, X = s.generateRealizationWithKF(N, u=U)

            if ci == 0:  # A special simple realization that is easy to track
                e = np.zeros((N, s.output_dim))
                e[0, :] = 1
                Y, X = s.generateRealizationWithKF(N, u=U, e=e)
                a = s.A
                apow = (a ** np.arange(len(Y))).T
                # import matplotlib.pyplot as plt
                # plt.figure()
                # ax = plt.gca()
                # ax.plot(X, label='X')
                # ax.plot(Y, label='Y')
                # ax.plot(apow, label='a^t')
                # ax.legend()
                # ax2 = ax.twinx()
                # ax2.plot(np.log10(np.abs(X)), linestyle='--', label='log10 |X|')
                # ax2.plot(np.log10(np.abs(Y)), linestyle='--', label='log10 |Y|')
                # ax2.plot(np.log10(np.abs(apow)), linestyle='--', label='log10 |a^t|')
                # ax2.legend(loc='lower right')
                # plt.show()

                Y = np.arange(1, N + 1)[:, np.newaxis]

            if s.input_dim > 0:
                YU = np.concatenate((Y, U), axis=1)
                UT, YUT = U.T, YU.T
            else:
                YU, YUT = Y, Y.T

            nx = s.state_dim
            sBW = copy.deepcopy(s)

            # Append one extra sample to Y to also get the 1-step prediction given the last actual sample of Y
            YFW = np.concatenate((Y, np.zeros_like(Y[0:1, :])), axis=0)
            UFW = (
                np.concatenate((U, np.zeros_like(U[0:1, :])), axis=0)
                if U is not None
                else None
            )
            allZpFW, allYpFW, allXpFW = s.predict(YFW, U=UFW)

            YBW = np.concatenate((np.flipud(Y), np.zeros_like(Y[0:2, :])), axis=0)
            UBW = (
                np.concatenate((np.flipud(U), np.zeros_like(U[0:2, :])), axis=0)
                if U is not None
                else None
            )
            allZpBW, allYpBW, allXpBW = sBW.predict(YBW, U=UBW)
            allZpBWFlip, allYpBWFlip, allXpBWFlip = (
                np.flipud(allZpBW),
                np.flipud(allYpBW),
                np.flipud(allXpBW),
            )

            shift_preds = False  # Must be False, True is to emulate old behavior that is like the behavior for one directional RNNs
            if shift_preds:
                allZpFW, allYpFW, allXpFW = (
                    allZpFW[:-1, :],
                    allYpFW[:-1, :],
                    allXpFW[:-1, :],
                )
                # We want the backward pass to see up to the exact same observation as was seen in the forward pass (that's how the Bidirectional RNN model works)
                # so we will append by two samples and remove the first two outputs
                allXp = np.concatenate((allXpFW, allXpBWFlip[:(-2), :]), axis=1)
                allYp = allYpFW + allYpBWFlip[:(-2), :]
                allZp = allZpFW + allZpBWFlip[:(-2), :]
            else:
                # For bidirectional, we expect shift_preds=False, in which case state at index i has seen inputs up to and including index i
                allZpFW, allYpFW, allXpFW = (
                    allZpFW[1:, :],
                    allYpFW[1:, :],
                    allXpFW[1:, :],
                )  # Drop first sample that is prediction given no observation
                allXp = np.concatenate((allXpFW, allXpBWFlip[1:(-1), :]), axis=1)
                allYp = allYpFW + allYpBWFlip[1:(-1), :]
                allZp = allZpFW + allZpBWFlip[1:(-1), :]

            # eagerly_flag_backup = set_global_tf_eagerly_flag(True)
            log_dir = "./logs"
            model2 = RNNModel(
                initLSSM=s,
                initLSSM_backward=sBW,
                bidirectional=True,
                # linear_cell=True,
                block_samples=block_samples,  # Making block samples the same as data length makes the results identical to LSSM, except for the edges
                batch_size=10,
                stateful=stateful,
                log_dir=log_dir,
            )
            allXp2, allYp2 = model2.predict(
                YUT, FT_in=UT, use_quick_method=True, shift_preds=shift_preds
            )[:2]

            outs = model2.predict_with_keras(
                (YUT, UT)
            )  # Just making sure that this can run too

            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(Y[:,:1], label='Y')
            plt.legend()
            plt.figure()
            plt.plot(allXp2[0:1,:].T, label='FW RNN XHat')
            plt.plot(allXp[:, 0:1], '--', label='FW LSSM XHat')
            [plt.axvline(x=x, color = 'r') for x in block_samples*(1+np.arange(round(N/block_samples), dtype=float))]
            plt.legend()
            plt.show()
            plt.figure()
            plt.plot(allXp2[nx:(nx+1),:].T, label='BW RNN XHat')
            plt.plot(allXp[:, nx:(nx+1)], '--', label='BW LSSM XHat')
            [plt.axvline(x=x, color = 'r') for x in block_samples*(1+np.arange(round(N/block_samples), dtype=float))]
            plt.legend()
            plt.show()
            plt.figure()
            plt.plot(allYp2[0:1,:].T, label='BW RNN YHat')
            plt.plot(allYp[:, 0:1], '--', label='BW LSSM YHat')
            [plt.axvline(x=x, color = 'r') for x in block_samples*(1+np.arange(round(N/block_samples), dtype=float))]
            plt.legend()
            plt.show()
            #"""

            try:
                # TO DO: fix discrepancies along block edges for the backward pass
                np.testing.assert_allclose(
                    allXpFW, allXp2[: allXpFW.shape[1], :].T, rtol=1e-3, atol=1e-6
                )
                np.testing.assert_allclose(
                    allXp[:, nx : (2 * nx)],
                    allXp2[nx : (2 * nx) :, :].T,
                    rtol=1e-3,
                    atol=1e-6,
                )
                np.testing.assert_allclose(allXp, allXp2.T, rtol=1e-3, atol=1e-6)
                np.testing.assert_allclose(allYp, allYp2.T, rtol=1e-3, atol=1e-6)
            except Exception as e:
                failInds.append(ci)
                failErrs.append(e)

        if len(failInds) > 0:
            raise (
                Exception(
                    "{} => {}/{} random systems (indices: {}) failed: \n{}".format(
                        self.id(), len(failInds), numTests, failInds, failErrs
                    )
                )
            )
        else:
            print(
                "{} => Ok: Tested with {} random systems, all were ok!".format(
                    self.id(), numTests
                )
            )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
