""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""The model used in DPAD"""
"""For mathematical description see DPADModelDoc.md"""

import copy
import io
import logging
import os
import re
import time
import warnings
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Some tools from our PSID library
from PSID import IPSID as LinIPSID
from PSID import LSSM
from PSID import PSID as LinPSID
from PSID import MatHelper as mh
from PSID.LSSM import genRandomGaussianNoise

from .RegressionModel import RegressionModel
from .RNNModel import RNNModel
from .tools.abstract_classes import PredictorModel
from .tools.file_tools import bytes_to_string, pickle_load, pickle_save
from .tools.LinearMapping import (
    LinearMappingSequence,
    getFlatRemoverMapping,
    getZScoreMapping,
)
from .tools.model_base_classes import ReconstructionInfo
from .tools.parse_tools import (
    extractNumberFromRegex,
    parseMethodCodeArgOptimizer,
    parseMethodCodeArgStepsAhead,
)
from .tools.plot import (
    checkIfAllExtsAlreadyExist,
    plotPredictionScatter,
    plotTimeSeriesPrediction,
    showOrSaveFig,
)
from .tools.tf_losses import (
    masked_CategoricalCrossentropy,
    masked_CC,
    masked_mse,
    masked_PoissonLL_loss,
    masked_R2,
)
from .tools.tf_tools import convertHistoryToDict, set_global_tf_eagerly_flag
from .tools.tools import (
    autoDetectSignalType,
    get_one_hot,
    get_trials_from_cat_data,
    getIsOk,
    isFlat,
)

logger = logging.getLogger(__name__)

INPUT_MSG = "Adding separate input signals is not supported in this version, but stay tuned for future DPAD versions that support it."
STEPS_AHEAD_MSG = "Setting steps ahead is not supported in this version, but stay tuned for future DPAD versions that wil support it."


def shift_ms_to_1s_series(
    Y, steps_ahead, missing_marker=None, time_first=True, verbose=True
):
    """Shifts multi step ahead predictions of the RNNModel from
        ====> time
        1-step: z^{0|-1}, z^{1| 0}, z^{2| 1}, z^{3| 2}, ...
        2-step: z^{1|-1}, z^{2| 0}, z^{3| 1}, z^{4| 2}, ...
        3-step: z^{2|-1}, z^{3| 0}, z^{4| 1}, z^{5| 2}, ...
        4-step: z^{3|-1}, z^{4| 0}, z^{5| 1}, z^{6| 2}, ...
    To:
        1-step: z^{0|-1}, z^{1| 0}, z^{2| 1}, z^{3| 2}, ...
        2-step:  nan    , z^{1|-1}, z^{2| 0}, z^{3| 1}, z^{4| 2}, ...
        3-step:  nan    ,  nan    , z^{2|-1}, z^{3| 0}, z^{4| 1}, z^{5| 2}, ...
        4-step:  nan    ,  nan    ,  nan    , z^{3|-1}, z^{4| 0}, z^{5| 1}, z^{6| 2}, ...
    So that in each sample, we have the prediction for the associate sample of the true
        signal, but from increasing older horizons
    Args:
        preds (list/tuple/np.array): predictions for all steps, or an np.array that is the prediction for one time step
        steps_ahead (list of int): list of steps ahead. if None will be taken as [1]
        missing_marker (np, optional): value that will be used to represent missing samples. Defaults to np.nan.
        time_first (bool, optional): if true, will take the first dim to be time.
                If False will take the 2nd dim to be time. Defaults to True.

    Returns:
        _type_: _description_
    """
    if Y is None:
        return None
    if missing_marker is None:
        missing_marker = np.nan
    if steps_ahead == None:
        steps_ahead = [1]
    Y_shifted = []
    for saInd, step_ahead in enumerate(steps_ahead):
        pred = Y[saInd] if isinstance(Y, (list, tuple)) else Y
        if pred is None:
            pred_shifted = pred
        elif step_ahead == 0:
            if verbose:
                logger.warning(
                    f"[TEMP] Ignoring shift for step_ahead=0 (treated exactly as step_ahead=1). Not expected for DPAD, unless bidirectional."
                )
            pred_shifted = pred
        elif step_ahead == 1:
            pred_shifted = pred
        elif step_ahead > 1:
            pred_shifted = missing_marker * np.ones_like(pred)
            if time_first:
                pred_shifted[(step_ahead - 1) :, ...] = pred[: (-step_ahead + 1), ...]
            else:
                pred_shifted[:, (step_ahead - 1) :, ...] = pred[
                    :, : (-step_ahead + 1), ...
                ]
        else:
            raise (Exception("step_ahead must be >= 1"))
        Y_shifted.append(pred_shifted)
    return tuple(Y_shifted)


def getLossLogStr(trueVals, predVals, steps, sigType, lossFuncs):
    if not isinstance(trueVals, (list, tuple)):
        trueVals = [trueVals]
    if not isinstance(predVals, (list, tuple)):
        predVals = [predVals]
    if steps is None:
        steps = [1]
    strs = []
    for ind in range(len(predVals)):
        trueVal = trueVals[ind % len(trueVals)]
        predVal = predVals[ind % len(predVals)]
        step = steps[ind % len(steps)]
        lossVals = [
            (
                lossFunc(trueVal, predVal.T)
                if sigType in ["cont", "count_process"]
                else lossFunc(trueVal, predVal.transpose([1, 0, 2]))
            )
            for lossFunc in lossFuncs
        ]
        strs.append(
            f"{step}-step: "
            + ", ".join(
                [
                    f"{lossFunc.__name__}={lossVal:.3g}"
                    for lossFunc, lossVal in zip(lossFuncs, lossVals)
                ]
            )
        )
    return "\n".join(strs)


def DPADModelSetTrainableParameters(
    base=None,
    fw=None,
    initial_state=None,
    model1=None,
    model2=None,
    model1_Cy=None,
    model2_Cz=None,
    model1_Cy_fw=None,
    model2_Cz_fw=None,
    model1_FCy=None,
    model1_FCy_fw=None,
):
    if model1 is not None:
        model1.setTrainableParameters(base=base, fw=fw, initial_state=initial_state)
    if model2 is not None:
        model2.setTrainableParameters(base=base, fw=fw, initial_state=initial_state)
    if base is not None:
        if model1_Cy is not None:
            model1_Cy.setTrainable(base)
        if model1_FCy is not None:
            model1_FCy.setTrainable(base)
        if model2_Cz is not None:
            model2_Cz.setTrainable(base)
    if fw is not None:
        if model1_Cy_fw is not None:
            model1_Cy_fw.setTrainable(fw)
        if model1_FCy_fw is not None:
            model1_FCy_fw.setTrainable(fw)
        if model2_Cz_fw is not None:
            model2_Cz_fw.setTrainable(fw)


class DPADModel(PredictorModel):
    """The main class implementing DPAD.
    x1(k+1) = A1( x1(k) ) + K1( y(k), u(k) )
    x2(k+1) = A2( x2(k) ) + K2( x1(k+1), y(k), u(k) )
    y(k)    = Cy( x1(k), x2(k), u(k) ) + ey_k
    z(k)    = Cz( x1(k), x2(k), u(k) ) + ez_k
    x(k) = [x1(k); x2(k)] => Latent state time series
    x1(k) => Latent states related to z
    x2(k) => Latent states unrelated to z
    """

    def __init__(
        self,
        block_samples=128,  # Number os timesteps in each training sample block
        batch_size=32,  # Each batch consists of this many blocks with block_samples time steps
        log_dir="",  # If not empty, will store tensorboard logs
        missing_marker=None,  # Values of z that are equal to this will not be used
        **kwargs,
    ):

        self.block_samples = block_samples
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.missing_marker = missing_marker

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def prepare_args(methodCode):
        """Parses a method code and preprares the arguments for the DPADModel constructor.

        Args:
            methodCode (string): DPAD method code string. For example can be "DPAD_uAKCzCy2HL128U"
                for a fully nonlinear model with 2 hidden layers of 128 units for each model parameter.

        Returns:
            kwargs: dict of arguments for the DPADModel constructor.
        """
        A1_args = {}
        K1_args = {}
        Cy1_args = {}
        Cz1_args = {}
        A2_args = {}
        K2_args = {}
        Cy2_args = {}
        Cz2_args = {}
        if "HL" in methodCode or "NonLin" in methodCode:
            regex = (
                r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+)HL(\d+)U"  # Cz1HL64U
            )
            if len(re.findall(regex, methodCode)) > 0:
                matches = re.finditer(regex, methodCode)
                for matchNum, match in enumerate(matches, start=1):
                    var_names, hidden_layers, hidden_units = match.groups()
                hidden_layers = int(hidden_layers)
                hidden_units = int(hidden_units)
            else:
                regex = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)NonLin"  # CzNonLin
                matches = re.finditer(regex, methodCode)
                for matchNum, match in enumerate(matches, start=1):
                    var_names = match.groups()
                hidden_layers = 3  # Default
                hidden_units = 64  # Default
            activation = "relu"
            NL_args = {
                "use_bias": True,
                "units": [hidden_units] * hidden_layers,
                "activation": activation,
            }
            if (
                var_names == ""
                or "A1" in var_names
                or ("A" in var_names and "A2" not in var_names)
            ):
                A1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "A2" in var_names
                or ("A" in var_names and "A1" not in var_names)
            ):
                A2_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "K1" in var_names
                or ("K" in var_names and "K2" not in var_names)
            ):
                K1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "K2" in var_names
                or ("K" in var_names and "K1" not in var_names)
            ):
                K2_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cy1" in var_names
                or ("Cy" in var_names and "Cy2" not in var_names)
            ):
                Cy1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cy2" in var_names
                or ("Cy" in var_names and "Cy1" not in var_names)
            ):
                Cy2_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cz1" in var_names
                or ("Cz" in var_names and "Cz2" not in var_names)
            ):
                Cz1_args = copy.copy(NL_args)
            if (
                var_names == ""
                or "Cz2" in var_names
                or ("Cz" in var_names and "Cz1" not in var_names)
            ):
                Cz2_args = copy.copy(NL_args)
        if (
            "OGInit" in methodCode
        ):  # Use initializers similar to the default keras LSTM initializers
            A1_args["kernel_initializer"] = "orthogonal"
            K1_args["kernel_initializer"] = "glorot_uniform"
            Cy1_args["kernel_initializer"] = "glorot_uniform"
            Cy1_args["kernel_initializer"] = "glorot_uniform"
            A2_args["kernel_initializer"] = "orthogonal"
            K2_args["kernel_initializer"] = "glorot_uniform"
            Cy2_args["kernel_initializer"] = "glorot_uniform"
            Cy2_args["kernel_initializer"] = "glorot_uniform"
        if "AKerIn0" in methodCode:  # Initialize A with zeros
            A1_args["kernel_initializer"] = "zeros"
            A2_args["kernel_initializer"] = "zeros"
        if "uAK" in methodCode:  # Unify A and K
            K1_args["unifiedAK"] = True
            K2_args["unifiedAK"] = True
        if "RGL" in methodCode:  # Regularize
            regex = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)RGLB?(\d+)?(Drop)?"  # _ARGL2_L1e5
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                var_names, norm_num, dropout = match.groups()
            # Param value
            lambdaVal = 0.01  # Default: 'l': 0.01
            regex = r"L(\d+)e([-+])?(\d+)"  # L1e-2
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                m, sgn, power = match.groups()
                if sgn is not None and sgn == "-":
                    power = -float(power)
                lambdaVal = float(m) * 10 ** float(power)
            RGL_args = {}
            if dropout is not None and dropout != "":  # Add dropout regularization
                RGL_args.update({"dropout_rate": lambdaVal})
            if norm_num is not None and norm_num != "":  # Add L1 or L2 regularization
                if norm_num in ["1", "2"]:
                    regularizer_name = "l{}".format(norm_num)
                else:
                    raise (Exception("Unsupported method code: {}".format(methodCode)))
                regularizer_args = {"l": lambdaVal}  # Default: 'l': 0.01
                RGL_args.update(
                    {
                        "kernel_regularizer_name": regularizer_name,
                        "kernel_regularizer_args": regularizer_args,
                    }
                )
                if "RGLB" in methodCode:  # Also regularize biases
                    # Reference for why usually biases don't need regularization:
                    # http://neuralnetworksanddeeplearning.com/chap3.html
                    RGL_args.update(
                        {
                            "bias_regularizer_name": regularizer_name,
                            "bias_regularizer_args": regularizer_args,
                        }
                    )
            if (
                var_names == ""
                or "A1" in var_names
                or ("A" in var_names and "A2" not in var_names)
            ):
                A1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "A2" in var_names
                or ("A" in var_names and "A1" not in var_names)
            ):
                A2_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "K1" in var_names
                or ("K" in var_names and "K2" not in var_names)
            ):
                K1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "K2" in var_names
                or ("K" in var_names and "K1" not in var_names)
            ):
                K2_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cy1" in var_names
                or ("Cy" in var_names and "Cy2" not in var_names)
            ):
                Cy1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cy2" in var_names
                or ("Cy" in var_names and "Cy1" not in var_names)
            ):
                Cy2_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cz1" in var_names
                or ("Cz" in var_names and "Cz2" not in var_names)
            ):
                Cz1_args.update(copy.deepcopy(RGL_args))
            if (
                var_names == ""
                or "Cz2" in var_names
                or ("Cz" in var_names and "Cz1" not in var_names)
            ):
                Cz2_args.update(copy.deepcopy(RGL_args))

        if "dummyA" in methodCode:  # Dummy A
            A1_args["dummy"] = True
            A2_args["dummy"] = True

        init_method = None
        if "init" in methodCode:  # Initialize
            regex = r"init(.+)"  #
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                initMethod = match.groups()
            if len(initMethod) > 0:
                init_method = initMethod[0]

        if "RTR" in methodCode:  # Retry initialization
            regex = r"RTR(\d+)"  # RTR2
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                init_attempts = match.groups()
            init_attempts = int(init_attempts[0])
        else:
            init_attempts = 1

        if "ErS" in methodCode:  # Early stopping
            regex = r"ErSV?(\d+)"  # ErS64
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                early_stopping_patience = match.groups()
                ErS_str = methodCode[match.span()[0] : match.span()[1]]
            early_stopping_patience = int(early_stopping_patience[0])
        else:
            early_stopping_patience = 16 # 16 is recommended for when early stopping is based on validation set. If based in train set, the recommended patience is 3.
            ErS_str = "ErSV"
        early_stopping_measure = "loss" if "ErSV" not in ErS_str else "val_loss"

        if "MinEp" in methodCode:  # Minimum epochs
            regex = r"MinEp(\d+)"  # MinEp150
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                start_from_epoch_rnn = match.groups()
            start_from_epoch_rnn = int(start_from_epoch_rnn[0])
        else:
            start_from_epoch_rnn = 0

        if "BtcS" in methodCode:  # The batch_size
            regex = r"BtcS(\d+)"  # BtcS1
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                batch_size = match.groups()
            batch_size = int(batch_size[0])
        else:
            batch_size = None

        lr_scheduler_name = None
        lr_scheduler_args = None

        optimizer_name = "Adam"  # default
        optimizer_args = None
        optimizer_infos, matches = parseMethodCodeArgOptimizer(methodCode)
        if len(optimizer_infos) > 0:
            optimizer_info = optimizer_infos[0]
            if "optimizer_name" in optimizer_info:
                optimizer_name = optimizer_info["optimizer_name"]
            if "optimizer_args" in optimizer_info:
                optimizer_args = optimizer_info["optimizer_args"]
            if "scheduler_name" in optimizer_info:
                lr_scheduler_name = optimizer_info["scheduler_name"]
            if "scheduler_args" in optimizer_info:
                lr_scheduler_args = optimizer_info["scheduler_args"]

        steps_ahead, steps_ahead_loss_weights, matches = parseMethodCodeArgStepsAhead(
            methodCode
        )

        model1_Cy_Full = (
            "FCy" in methodCode or "FCyCz" in methodCode or "FCzCy" in methodCode
        )
        model2_Cz_Full = (
            "FCz" in methodCode or "FCyCz" in methodCode or "FCzCy" in methodCode
        )
        linear_cell = "LinCell" in methodCode
        LSTM_cell = "LSTM" in methodCode
        bidirectional = "xSmth" in methodCode or "bidir" in methodCode
        allow_nonzero_Cz2 = "Cz20" not in methodCode
        has_Dyz = "Dyz" in methodCode
        skip_Cy = "skipCy" in methodCode
        zscore_inputs = "nzs" not in methodCode

        kwargs = {
            "A1_args": A1_args,
            "K1_args": K1_args,
            "Cy1_args": Cy1_args,
            "Cz1_args": Cz1_args,
            "A2_args": A2_args,
            "K2_args": K2_args,
            "Cy2_args": Cy2_args,
            "Cz2_args": Cz2_args,
            "init_method": init_method,
            "init_attempts": init_attempts,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_measure": early_stopping_measure,
            "start_from_epoch_rnn": start_from_epoch_rnn,
            "model1_Cy_Full": model1_Cy_Full,
            "model2_Cz_Full": model2_Cz_Full,
            "linear_cell": linear_cell,
            "LSTM_cell": LSTM_cell,
            "bidirectional": bidirectional,
            "allow_nonzero_Cz2": allow_nonzero_Cz2,
            "has_Dyz": has_Dyz,
            "skip_Cy": skip_Cy,
            "steps_ahead": steps_ahead,  # List of ints (None take as [1]), indicating the number of steps ahead to generate from the model (used to construct training loss and predictions
            "steps_ahead_loss_weights": steps_ahead_loss_weights,  # Weight of each step ahead prediction in loss. If None, will give all steps ahead equal weight of 1.
            "zscore_inputs": zscore_inputs,
            "optimizer_name": optimizer_name,
            "optimizer_args": optimizer_args,
            "lr_scheduler_name": lr_scheduler_name,
            "lr_scheduler_args": lr_scheduler_args,
        }
        if A1_args == A2_args:
            kwargs["A_args"] = A1_args
            del kwargs["A1_args"], kwargs["A2_args"]
        if K1_args == K2_args:
            kwargs["K_args"] = K1_args
            del kwargs["K1_args"], kwargs["K2_args"]
        if Cy1_args == Cy2_args:
            kwargs["Cy_args"] = Cy1_args
            del kwargs["Cy1_args"], kwargs["Cy2_args"]
        if Cz1_args == Cz2_args:
            kwargs["Cz_args"] = Cz1_args
            del kwargs["Cz1_args"], kwargs["Cz2_args"]
        return kwargs

    def add_default_param_args(
        self,
        A1_args={},
        K1_args={},
        Cy1_args={},
        Cz1_args={},
        A2_args={},
        K2_args={},
        Cy2_args={},
        Cz2_args={},
        yDist=None,
        zDist=None,
    ):
        LinArgs = {
            "units": [],
            "use_bias": False,
            "activation": "linear",
            "output_activation": "linear",
        }
        for f, v in LinArgs.items():
            if f not in A1_args:
                A1_args[f] = v
            if f not in K1_args:
                K1_args[f] = v
            if f not in Cy1_args:
                Cy1_args[f] = v
            if f not in Cz1_args:
                Cz1_args[f] = v
            if f not in A2_args:
                A2_args[f] = v
            if f not in K2_args:
                K2_args[f] = v
            if f not in Cy2_args:
                Cy2_args[f] = v
            if f not in Cz2_args:
                Cz2_args[f] = v

        if yDist == "poisson":
            Cy1_args["out_dist"] = "poisson"
            Cy1_args["output_activation"] = "exponential"
            Cy2_args["out_dist"] = "poisson"
            Cy2_args["output_activation"] = "exponential"

        if zDist == "poisson":
            Cz1_args["out_dist"] = "poisson"
            Cz1_args["output_activation"] = "exponential"
            Cz2_args["out_dist"] = "poisson"
            Cz2_args["output_activation"] = "exponential"

        if "unifiedAK" not in K1_args:
            K1_args["unifiedAK"] = False
        if "unifiedAK" not in K2_args:
            K2_args["unifiedAK"] = False
        return (
            A1_args,
            K1_args,
            Cy1_args,
            Cz1_args,
            A2_args,
            K2_args,
            Cy2_args,
            Cz2_args,
        )

    def get_model_steps_ahead(self, steps_ahead=None, steps_ahead_loss_weights=None):
        if steps_ahead is not None:
            maxStepsAhead = np.max(steps_ahead)
            minStepAhead = 1 if 0 not in steps_ahead else 0
            steps_ahead_model1 = list(range(minStepAhead, 1 + maxStepsAhead))
            model1_orig_step_inds = [
                np.where(np.array(steps_ahead_model1) == step_ahead)[0][0]
                for step_ahead in steps_ahead
            ]
            steps_ahead_loss_weights_model1 = [
                0.0 for saInd in range(len(steps_ahead_model1))
            ]
            for saInd in range(len(steps_ahead)):
                if steps_ahead_loss_weights is not None:
                    steps_ahead_loss_weights_model1[model1_orig_step_inds[saInd]] = (
                        steps_ahead_loss_weights[saInd]
                    )
                else:
                    steps_ahead_loss_weights_model1[model1_orig_step_inds[saInd]] = 1.0
        else:
            steps_ahead = [
                1 if not self.bidirectional else 0
            ]  # Default is 1 step ahead
            steps_ahead_model1 = steps_ahead
            steps_ahead_loss_weights_model1 = steps_ahead_loss_weights
            model1_orig_step_inds = [0]
        return (
            steps_ahead,
            steps_ahead_loss_weights,
            steps_ahead_model1,
            steps_ahead_loss_weights_model1,
            model1_orig_step_inds,
        )

    def prep_observation_for_training(self, Y, YType):
        """Prepares the output distribution depending of signal type, a version of the output loss function,
        and appropriated shaped ground truth signal for logging

        Args:
            Y ([type]): [description]
            YType ([type]): [description]

        Returns:
            [type]: [description]
        """
        if Y is not None:
            isOkY = getIsOk(Y, self.missing_marker)
        else:
            YTrue = Y
        if YType == "cat":
            yDist = None
            YLossFuncs = [
                masked_CategoricalCrossentropy(self.missing_marker),
            ]
            if Y is not None:
                YClasses = np.unique(Y[:, np.all(isOkY, axis=0)])
                YTrue = np.ones((Y.shape[1], Y.shape[0], len(YClasses)), dtype=int) * (
                    int(self.missing_marker) if self.missing_marker is not None else -1
                )
                for yi in range(Y.shape[0]):
                    YTrueThis = get_one_hot(
                        Y[yi, isOkY[yi, :]][:, np.newaxis], len(YClasses)
                    )
                    YTrue[isOkY[yi, :], yi, :] = YTrueThis[:, 0, :]
        elif YType == "count_process":
            yDist = "poisson"
            YLossFuncs = [masked_PoissonLL_loss(self.missing_marker)]
            if Y is not None:
                YTrue = Y.T
        else:
            yDist = None
            YLossFuncs = [
                masked_mse(self.missing_marker),
                masked_R2(self.missing_marker),
                masked_CC(self.missing_marker),
            ]
            if Y is not None:
                YTrue = Y.T
        return YLossFuncs, YTrue, yDist

    def get_input_prep_map(
        self,
        Y,
        remove_flat_dims=True,
        zscore_inputs=True,
        zscore_per_dim=True,
        signal_name="",
    ):
        YThis = Y
        map = LinearMappingSequence()
        # Detect and remove flat data dimensions
        if remove_flat_dims:
            thisMap = getFlatRemoverMapping(YThis, signal_name, axis=1)
            if thisMap is not None:
                map.append(thisMap)
                YThis = thisMap.apply(YThis)
        if zscore_inputs:
            thisMap = getZScoreMapping(
                YThis, signal_name, axis=1, zscore_per_dim=zscore_per_dim
            )
            if thisMap is not None:
                map.append(thisMap)
            # YThis = thisMap.apply(YThis)
        return map

    def setTrainableParameters(self, base=None, fw=None, initial_state=None):
        DPADModelSetTrainableParameters(
            base=base,
            fw=fw,
            initial_state=initial_state,
            model1=self.model1 if hasattr(self, "model1") else None,
            model2=self.model2 if hasattr(self, "model2") else None,
            model1_Cy=self.model1_Cy if hasattr(self, "model1_Cy") else None,
            model2_Cz=self.model2_Cz if hasattr(self, "model2_Cz") else None,
            model1_Cy_fw=self.model1_Cy_fw if hasattr(self, "model1_Cy_fw") else None,
            model2_Cz_fw=self.model2_Cz_fw if hasattr(self, "model2_Cz_fw") else None,
        )

    def finetune(self, Y, Z=None, U=None, **kw_args):
        """Finetunes a learned DPAD model by starting from that model and continuing optimization

        Args:
            Y (_type_): _description_
            Z (_type_, optional): _description_. Defaults to None.
            U (_type_, optional): _description_. Defaults to None.
        """
        # Defaults that will be used even if kw_args has a value for them (value in kw_args will be discarded)
        forced_defaults = {
            "nx": self.nx,
            "n1": self.n1,
            "batch_size": self.batch_size,  # If not none, will set the batch_size of the DPADModel to this value
            "init_model": self,  # Will initialize by provided method
            "init_method": None,  # Will initialize by the provided method
            "regression_init_method": None,  # Will initialize regressions with the provided method
            "init_attempts": 1,  # Will retry with different (random) initializations at least this many times
            "max_attempts": 1,  # Max refit attempts in case a model leads to nan loss
            "linear_cell": (
                self.linear_cell if hasattr(self, "linear_cell") else False
            ),  # If true, will use a linear cell instead of general regression cell (JUST FOR TESTS)
            "LSTM_cell": (
                self.LSTM_cell if hasattr(self, "LSTM_cell") else False
            ),  # If true, will use an LSTM cell in the RNNs instead of general regression cell
            "allow_nonzero_Cz2": (
                self.allow_nonzero_Cz2
                if hasattr(self, "allow_nonzero_Cz2")
                else (self.n1 == self.nx)
            ),  # If true, will use the states in the second stage to add to the decoding of behavior.
            # This usually doesn't help and can even cause some overfitting for very large n2.
            "has_Dyz": (
                self.has_Dyz if hasattr(self, "has_Dyz") else False
            ),  # If true, will also fit a direct regression from Y to Z
            "model1_Cy_Full": self.model1_Cy_Full,
            "model2_Cz_Full": self.model2_Cz_Full,
            "skip_Cy": (
                self.skip_Cy if hasattr(self, "linear_cell") else False
            ),  # If true and only stage 1 (n1 >= nx), will not learn Cy (model will not have neural self-prediction ability)
            "use_existing_prep_models": True,
            "clear_graph": False,  # If true will wipe the tf session before starting, so that variables names don't get numbers at the end and mem is preserved
            "YType": self.YType,
            "ZType": self.ZType,
            "UType": self.UType if hasattr(self, "UType") else None,
            "A_args": self.A_args,
            "K_args": self.K_args,
            "Cy_args": self.Cy_args,
            "Cz_args": self.Cz_args,  # Both stage 1 and 2 params
            "A1_args": self.A1_args,
            "K1_args": self.K1_args,
            "Cy1_args": self.Cy1_args,
            "Cz1_args": self.Cz1_args,  # Stage 1 params
            "A2_args": self.A2_args,
            "K2_args": self.K2_args,
            "Cy2_args": self.Cy2_args,
            "Cz2_args": self.Cz2_args,  # Stage 2 params
            "optimizer_name": (
                self.optimizer_name if hasattr(self, "optimizer_name") else "Adam"
            ),  # Name of optimizer
            "optimizer_args": (
                self.optimizer_args if hasattr(self, "optimizer_args") else None
            ),  # Dict of arguments for the optimizer
            "lr_scheduler_name": (
                self.lr_scheduler_name if hasattr(self, "lr_scheduler_name") else None
            ),  # Name of learning rate scheduler
            "lr_scheduler_args": (
                self.lr_scheduler_args if hasattr(self, "lr_scheduler_args") else None
            ),  # Dict of arguments for the learning rate scheduler
        }
        kw_args.update(forced_defaults)
        # Soft defaults for which the value from kw_args (if available) will instead be used
        soft_defaults = {
            "enable_forward_pred": (
                self.enable_forward_pred
                if hasattr(self, "enable_forward_pred")
                else False
            ),  # If True, will enable forward prediction by having a separate set of Afw,Kfw,Cfw (if needed)
            "steps_ahead": self.steps_ahead,  # List of ints (None take as [1]), indicating the number of steps ahead to generate from the model (used to construct training loss and predictions
            "steps_ahead_loss_weights": (
                self.steps_ahead_loss_weights
                if hasattr(self, "steps_ahead_loss_weights")
                else None
            ),  # Weight of each step ahead prediction in loss. If None, will give all steps ahead equal weight of 1.
            "throw_on_fail": False,
        }
        soft_defaults.update(kw_args)
        logger.info(f"DPAD.finetune args: {soft_defaults}")
        self.fit(Y, Z=Z, U=U, **soft_defaults)

    def fit(
        self,
        Y,
        Z=None,
        U=None,
        nx=None,
        n1=None,
        epochs=2500,  # Max number of epochs to go over the whole training data
        batch_size=None,  # If not none, will set the batch_size of the DPADModel to this value
        init_model=None,  # Will initialize by provided method
        init_method=None,  # Will initialize by the provided method
        regression_init_method=None,  # Will initialize regressions with the provided method
        init_attempts=1,  # Will retry with different (random) initializations at least this many times
        max_attempts=10,  # Max refit attempts in case a model leads to nan loss
        throw_on_fail=True,  # If true, will raise exception if learned RNN is unstable
        early_stopping_patience=16,  # Will stop numerical optimization early any time this many epochs do not bring an improvement in the loss
        early_stopping_measure="val_loss",  # The measure to use for early stopping
        start_from_epoch_rnn=0,  # Minimum epochs to do before early stopping is considered for the rnn fitting (model1, model2) optimizations
        start_from_epoch_reg=0,  # Minimum epochs to do before early stopping is considered for the regression (model1_Cy, model2_Cz) fitting optimizations
        create_val_from_training=False,  # If True, will create the validation data by cutting it off of the training data
        validation_set_ratio=0.2,  # Ratio of training data to cut off to make the validation data
        Y_validation=None,  # if provided will use to compute loss on validation (unless create_val_from_training=True)
        Z_validation=None,  # if provided will use to compute loss on validation (unless create_val_from_training=True)
        U_validation=None,  # if provided will use to compute loss on validation (unless create_val_from_training=True)
        true_model=None,
        linear_cell=False,  # If true, will use a linear cell instead of general regression cell (JUST FOR TESTS)
        LSTM_cell=False,  # If true, will use an LSTM cell in the RNNs instead of general regression cell
        bidirectional=False,  # If true, will use bidirectional RNNs, resulting in a non-causal inference
        allow_nonzero_Cz2=True,  # If true, will use the states in the second stage to add to the decoding of behavior.
        # This usually doesn't help and can even cause some overfitting for very large n2.
        has_Dyz=False,  # If true, will also fit a direct regression from Y to Z
        model1_Cy_Full=False,
        model2_Cz_Full=True,
        skip_Cy=False,  # If true and only stage 1 (n1 >= nx), will not learn Cy (model will not have neural self-prediction ability)
        remove_flat_dims=True,  # (default: True) Fitting may not work well if data has flat dimensions, so this should be true so that flat dimensions are handled separately in a preprocessing step
        zscore_inputs=True,  # (default: True) Fitting will typically be more robust if inputs are z-scored
        zscore_per_dim=False,  # If true, will z-score inputs per dimension. This is the default, but can be disabled if you want the method to focus on some dims (with larger magnitudes) more than others.
        use_existing_prep_models=False,  # For finetuning, will use existing preprocessing models
        trainableParams=None,  # Specifies the parameters that are learnable (useful in finetuning)
        clear_graph=True,  # If true will wipe the tf session before starting, so that variables names don't get numbers at the end and mem is preserved
        verbose=True,  # If True will print more logs
        save_logs=True,  # If true, will return fit logs for all models
        skip_predictions=False,  # JUST FOR TESTS to make them faster
        YType=None,
        ZType=None,
        UType=None,
        time_first=False,
        enable_forward_pred=True,  # If True, will enable forward prediction by having a separate set of Afw,Kfw,Cfw (if needed)
        steps_ahead=None,  # List of ints (None take as [1]), indicating the number of steps ahead to generate from the model (used to construct training loss and predictions
        steps_ahead_loss_weights=None,  # Weight of each step ahead prediction in loss. If None, will give all steps ahead equal weight of 1.
        A_args={},
        K_args={},
        Cy_args={},
        Cz_args={},  # Both stage 1 and 2 params
        A1_args=None,
        K1_args=None,
        Cy1_args=None,
        Cz1_args=None,  # Stage 1 params
        A2_args=None,
        K2_args=None,
        Cy2_args=None,
        Cz2_args=None,  # Stage 2 params
        optimizer_name="Adam",  # Name of optimizer
        optimizer_args=None,  # Dict of arguments for the optimizer
        lr_scheduler_name=None,  # Name of learning rate scheduler class (e.g., ExponentialDecay)
        lr_scheduler_args=None,  # Dict of arguments for the learning rate scheduler
    ):
        eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        if clear_graph:
            tf.keras.backend.clear_session()

        if time_first:
            raise (
                Exception(
                    "Time being the first dim is not supported for DPAD yet. Please transpose your data to have time as the 2nd dimension."
                )
            )

        if U is not None:
            raise (Exception(INPUT_MSG))

        # Prepare validation data
        if "val" in early_stopping_measure:
            logger.info(
                f'Early stopping measure is "{early_stopping_measure}". Includes validation loss, so we will ignore any provided validation data and creating validation data fom the training data to avoid overfitting'
            )
            create_val_from_training = True
            Y_validation, Z_validation, U_validation = (
                None,
                None,
                None,
            )  # Ignore any provided validation data

        isTrialBased = isinstance(Y, (list, tuple))
        if isTrialBased:
            # Trial based learning
            segment_length = [YThis.shape[1] for YThis in Y]
            if np.max(segment_length) != np.min(segment_length):
                raise (
                    Exception(
                        "Trials must be same length for trial-based learning (but they spanned {} to {} samples)".format(
                            np.min(segment_length), np.max(segment_length)
                        )
                    )
                )
            self.block_samples = segment_length[0]  # Set block samples to trial length
            self.stateful = False
            self.learn_initial_state = True

            # Concatenate data
            def convertToCat(data):
                if data is None:
                    return data
                dataArray = np.array(data)
                return dataArray.transpose(1, 2, 0).reshape(
                    dataArray.shape[1],
                    dataArray.shape[0] * dataArray.shape[2],
                    order="F",
                )

            Y = convertToCat(Y)
            Z = convertToCat(Z)
            U = convertToCat(U)
            Y_validation = convertToCat(Y_validation)
            Z_validation = convertToCat(Z_validation)
            U_validation = convertToCat(U_validation)
        else:
            self.stateful = True
            self.learn_initial_state = False

        Ndat = Y.shape[1]
        if create_val_from_training:
            if isTrialBased:  # Train/validation sets must be a multiple of trial length
                test_sample_1 = (
                    int(
                        np.floor(Ndat / self.block_samples * (1 - validation_set_ratio))
                    )
                    * self.block_samples
                )
            else:
                test_sample_1 = int(np.floor(Ndat * (1 - validation_set_ratio)))
            Y_validation = Y[:, test_sample_1:]
            Y = Y[
                :, 0:test_sample_1
            ]  # Remove the validation data from the training data
            if Z is not None:
                Z_validation = Z[:, test_sample_1:]
                Z = Z[
                    :, 0:test_sample_1
                ]  # Remove the validation data from the training data
            if U is not None:
                U_validation = U[:, test_sample_1:]
                U = U[
                    :, 0:test_sample_1
                ]  # Remove the validation data from the training data
            Ndat_for_batch_size = np.min(
                (Ndat, Y_validation.shape[1])
            )  # Make sure batch size is not too large for validation data
        else:
            Ndat_for_batch_size = Ndat

        if batch_size is not None:
            self.batch_size = batch_size

        num_batch = int(
            np.ceil((Ndat_for_batch_size - 1) / self.block_samples / self.batch_size)
        )
        if num_batch < 2:
            num_batch = 1
        self.batch_size = int(
            np.floor((Ndat_for_batch_size - 1) / self.block_samples / num_batch)
        )
        if self.batch_size < 1:
            raise Exception("Not enough samples for model fitting")

        if YType is None and Y is not None:  # Auto detect signal types
            YType = autoDetectSignalType(Y)
        if UType is None and U is not None:  # Auto detect signal types
            UType = autoDetectSignalType(U)
        if Z is not None:
            if ZType is None:
                ZType = autoDetectSignalType(Z)
            ZLossFuncs, ZTrue, zDist = self.prep_observation_for_training(Z, ZType)
        else:
            ZTrue, zDist = None, None

        YLossFuncs, YTrue, yDist = self.prep_observation_for_training(Y, YType)

        if use_existing_prep_models:
            YPrepMap = self.YPrepMap if hasattr(self, "YPrepMap") else None
            ZPrepMap = self.ZPrepMap if hasattr(self, "ZPrepMap") else None
            UPrepMap = self.UPrepMap if hasattr(self, "UPrepMap") else None
            remove_flat_dims = (
                self.remove_flat_dims if hasattr(self, "remove_flat_dims") else False
            )
            zscore_inputs = (
                self.zscore_inputs if hasattr(self, "zscore_inputs") else False
            )
            zscore_inputs = (
                self.zscore_inputs if hasattr(self, "zscore_inputs") else False
            )
        else:
            # Detect and remove flat dimensions of Y and U since they have no info (otherwise RNN fitting would not work well due to gradient explosion issues)
            YPrepMap, UPrepMap, ZPrepMap = None, None, None
            if remove_flat_dims or zscore_inputs:
                YPrepMap = self.get_input_prep_map(
                    Y,
                    signal_name="Y",
                    remove_flat_dims=remove_flat_dims,
                    zscore_inputs=zscore_inputs and YType == "cont",
                    zscore_per_dim=zscore_per_dim,
                )
                UPrepMap = self.get_input_prep_map(
                    U,
                    signal_name="U",
                    remove_flat_dims=remove_flat_dims,
                    zscore_inputs=zscore_inputs and UType == "cont",
                    zscore_per_dim=zscore_per_dim,
                )
                ZPrepMap = self.get_input_prep_map(
                    Z,
                    signal_name="Z",
                    remove_flat_dims=remove_flat_dims,
                    zscore_inputs=zscore_inputs and ZType == "cont",
                    zscore_per_dim=zscore_per_dim,
                )

        if YPrepMap is not None:
            if Y is not None:
                Y = YPrepMap.apply(Y)
            if YTrue is not None:
                YTrue = YPrepMap.apply(YTrue.T).T
            if Y_validation is not None:
                Y_validation = YPrepMap.apply(Y_validation)
        if ZPrepMap is not None:
            if Z is not None:
                Z = ZPrepMap.apply(Z)
            if ZTrue is not None:
                ZTrue = ZPrepMap.apply(ZTrue.T).T
            if Z_validation is not None:
                Z_validation = ZPrepMap.apply(Z_validation)
        if UPrepMap is not None:
            if U is not None:
                U = UPrepMap.apply(U)
            if U_validation is not None:
                U_validation = UPrepMap.apply(U_validation)

        if nx is None:
            nx = self.nx
        if n1 is None:
            if hasattr(self, "n1") and self.n1 is not None:
                n1 = self.n1
            else:
                n1 = 0  # By default only use stage 2

        ny, Ndat = Y.shape[0], Y.shape[1]
        if Z is not None:
            nz, NdatZ = Z.shape[0], Z.shape[1]
        else:
            nz = 0
            n1 = 0

        if U is not None:
            nu = U.shape[0]
            UT = U.T
        else:
            nu = 0
            UT = None

        if n1 > nx:
            n1 = nx  # Max possible n1 value
        n2 = nx - n1

        if n1 == 0 or model2_Cz_Full:
            allow_nonzero_Cz2 = True  # Must use Cz2 if n1=0

        isOkY = getIsOk(Y, self.missing_marker)
        isOkZ = getIsOk(Z, self.missing_marker)
        isOkU = getIsOk(U, self.missing_marker)

        if A1_args is None:
            A1_args = A_args
        if A2_args is None:
            A2_args = A_args
        if K1_args is None:
            K1_args = K_args
        if K2_args is None:
            K2_args = K_args
        if Cy1_args is None:
            Cy1_args = Cy_args
        if Cy2_args is None:
            Cy2_args = Cy_args
        if Cz1_args is None:
            Cz1_args = Cz_args
        if Cz2_args is None:
            Cz2_args = Cz_args

        (
            A1_args,
            K1_args,
            Cy1_args,
            Cz1_args,
            A2_args,
            K2_args,
            Cy2_args,
            Cz2_args,
        ) = self.add_default_param_args(
            A1_args,
            K1_args,
            Cy1_args,
            Cz1_args,
            A2_args,
            K2_args,
            Cy2_args,
            Cz2_args,
            yDist,
            zDist,
        )

        if YType == "count_process":
            Cy1_args["use_bias"] = True
            Cy2_args["use_bias"] = True
            Cy_args["use_bias"] = True
        elif YType == "cat":
            YClasses = np.unique(Y)  # Get the y classes
            Cy1_args["num_classes"] = len(YClasses)
            Cy2_args["num_classes"] = len(YClasses)
            Cy_args["num_classes"] = len(YClasses)
            Cy1_args["use_bias"] = True
            Cy2_args["use_bias"] = True
            Cy_args["use_bias"] = True
            self.YClasses = YClasses

        if ZType == "count_process":
            Cz1_args["use_bias"] = True
            Cz2_args["use_bias"] = True
            Cz_args["use_bias"] = True
        elif ZType == "cat":
            ZClasses = np.unique(Z)  # Get the z classes
            Cz1_args["num_classes"] = len(ZClasses)
            Cz2_args["num_classes"] = len(ZClasses)
            Cz_args["num_classes"] = len(ZClasses)
            Cz1_args["use_bias"] = True
            Cz2_args["use_bias"] = True
            Cz_args["use_bias"] = True
            self.ZClasses = ZClasses

        # Store description of the network in the object
        self.YType = YType
        self.ZType = ZType

        self.Cz_args = copy.deepcopy(Cz_args)
        self.Cy_args = copy.deepcopy(Cy_args)
        self.A_args = copy.deepcopy(A_args)
        self.K_args = copy.deepcopy(K_args)
        self.Cz1_args = copy.deepcopy(Cz1_args)
        self.Cy1_args = copy.deepcopy(Cy1_args)
        self.A1_args = copy.deepcopy(A1_args)
        self.K1_args = copy.deepcopy(K1_args)
        self.Cz2_args = copy.deepcopy(Cz2_args)
        self.Cy2_args = copy.deepcopy(Cy2_args)
        self.A2_args = copy.deepcopy(A2_args)
        self.K2_args = copy.deepcopy(K2_args)
        self.linear_cell = linear_cell
        self.LSTM_cell = LSTM_cell
        self.bidirectional = bidirectional
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_args = lr_scheduler_args
        self.enable_forward_pred = enable_forward_pred
        self.steps_ahead = steps_ahead
        self.steps_ahead_loss_weights = steps_ahead_loss_weights
        self.remove_flat_dims = remove_flat_dims
        self.zscore_inputs = zscore_inputs

        self.ny = ny
        self.nz = nz
        self.nu = nu
        self.nx = nx
        self.n1 = n1
        self.n2 = n2
        # self.state_dim = self.nx
        self.model1_Cy_Full = model1_Cy_Full
        self.model2_Cz_Full = model2_Cz_Full
        self.allow_nonzero_Cz2 = allow_nonzero_Cz2
        self.has_Dyz = has_Dyz
        self.skip_Cy = skip_Cy

        # Prepare the network
        (
            model1,
            model1_Cy,
            model1_Cy_fw,
            model2,
            model2_Cz,
            model2_Cz_fw,
            model1_FCy,
            model1_FCy_fw,
        ) = self.build_models()
        if trainableParams is not None:
            DPADModelSetTrainableParameters(
                model1=model1,
                model1_Cy=model1_Cy,
                model1_Cy_fw=model1_Cy_fw,
                model2=model2,
                model2_Cz=model2_Cz,
                model2_Cz_fw=model2_Cz_fw,
                model1_FCy=model1_FCy,
                model1_FCy_fw=model1_FCy_fw,
                **trainableParams,
            )

        # For comparison or initialization
        if init_model is not None:
            sInit0 = init_model
        elif init_method is not None:
            if isTrialBased:
                YA = list(
                    get_trials_from_cat_data(Y.T, self.block_samples).transpose(0, 2, 1)
                )
                ZA = (
                    list(
                        get_trials_from_cat_data(Z.T, self.block_samples).transpose(
                            0, 2, 1
                        )
                    )
                    if Z is not None
                    else None
                )
                UA = (
                    list(
                        get_trials_from_cat_data(U.T, self.block_samples).transpose(
                            0, 2, 1
                        )
                    )
                    if U is not None
                    else None
                )
            else:
                YA = Y
                ZA = Z
                UA = U
            PSIDFitFunc = LinIPSID
            PSIDFitFuncUArgs = {}
            if init_method == "PSID":
                sInit0 = PSIDFitFunc(
                    YA,
                    ZA,
                    nx=nx,
                    n1=n1,
                    i=10,
                    time_first=False,
                    missing_marker=self.missing_marker,
                    **PSIDFitFuncUArgs,
                )
            elif init_method in ["SID", "SID_RR"]:
                sInit0 = PSIDFitFunc(
                    YA,
                    ZA,
                    nx=nx,
                    n1=0,
                    i=10,
                    time_first=False,
                    missing_marker=self.missing_marker,
                    **PSIDFitFuncUArgs,
                )
            elif init_method in ["True"]:
                sInit0 = true_model
            else:
                raise (Exception("Not supported!"))
        else:
            sInit0 = None
        if sInit0 is not None:
            if verbose and not skip_predictions:
                # """
                # Compute z-decoding MSE:
                preds = sInit0.predict(Y.T, U=UT)
                steps_ahead_sInit0 = (
                    sInit0.steps_ahead
                    if (
                        hasattr(sInit0, "steps_ahead")
                        and sInit0.steps_ahead is not None
                    )
                    else [1]
                )
                allZp_steps = preds[: len(steps_ahead_sInit0)]
                allYp_steps = preds[
                    len(steps_ahead_sInit0) : 2 * len(steps_ahead_sInit0)
                ]
                allXp_steps = preds[
                    len(steps_ahead_sInit0) : 2 * len(steps_ahead_sInit0)
                ]
                allZp, allYp, allXp = allZp_steps[0], allYp_steps[0], allXp_steps[0]
                if Z is not None:
                    logger.info(
                        "Init Z pred in training data: "
                        + getLossLogStr(ZTrue, allZp, None, ZType, ZLossFuncs)
                    )
                logger.info(
                    "Init Y pred in training data: "
                    + getLossLogStr(Y, allYp, None, YType, YLossFuncs)
                )
                # """
            if isinstance(sInit0, (DPADModel)):
                sInit = sInit0
            elif (
                ("activation" in A1_args and A1_args["activation"] != "linear")
                or ("activation" in K1_args and K1_args["activation"] != "linear")
                or ("activation" in Cy1_args and Cy1_args["activation"] != "linear")
                or ("activation" in Cz1_args and Cz1_args["activation"] != "linear")
                or ("activation" in A2_args and A2_args["activation"] != "linear")
                or ("activation" in K2_args and K2_args["activation"] != "linear")
                or ("activation" in Cy2_args and Cy2_args["activation"] != "linear")
                or ("activation" in Cz2_args and Cz2_args["activation"] != "linear")
            ):
                # Has nonlinearity
                logger.info(
                    "Trying to convert the init model to the computation graph of the final model"
                )
                # Generate new linear data from the linear model
                YLin, XLin, ZLin = sInit0.generateRealization(
                    N=10000, u=UT, return_z=True
                )
                sInit = DPADModel(
                    nx=nx,
                    n1=n1,
                    nu=nu,
                    block_samples=self.block_samples,
                    batch_size=self.batch_size,
                )
                sInit.fit(
                    Y=YLin.T,
                    Z=ZLin.T,
                    U=UT,
                    time_first=False,
                    A1_args=A1_args,
                    K1_args=K1_args,
                    Cy1_args=Cy1_args,
                    Cz1_args=Cz1_args,
                    A2_args=A2_args,
                    K2_args=K2_args,
                    Cy2_args=Cy2_args,
                    Cz2_args=Cz2_args,
                    enable_forward_pred=enable_forward_pred,
                    steps_ahead=steps_ahead,
                )
                self.initE = sInit
            else:
                sInit = DPADModel(nx=nx, n1=n1, nu=nu)
                self.initE = sInit.setToLSSM(
                    sInit0,
                    model1_Cy_Full=model1_Cy_Full,
                    model2_Cz_Full=model2_Cz_Full,
                    allow_nonzero_Cz2=allow_nonzero_Cz2,
                    YType=YType,
                    ZType=ZType,
                    A1_args=A1_args,
                    K1_args=K1_args,
                    Cy1_args=Cy1_args,
                    Cz1_args=Cz1_args,
                    A2_args=A2_args,
                    K2_args=K2_args,
                    Cy2_args=Cy2_args,
                    Cz2_args=Cz2_args,
                    enable_forward_pred=enable_forward_pred,
                    steps_ahead=steps_ahead,
                    ignore_Zero_A_KC_topRight=True,
                    ignore_Zero_A_topRight=True,
                )

            # Initialize models
            if model1 is not None:
                if verbose:
                    logger.info(f'Initializing RNN1 with method "{init_method}"')
                model1.set_cell_weights(sInit.model1.get_cell_weights())
                if epochs == 0:
                    model1.set_batch_size(1)
            if model1_Cy is not None:
                if verbose:
                    logger.info(f'Initializing model1_Cy with method "{init_method}"')
                model1_Cy.model.set_weights(sInit.model1_Cy.model.get_weights())
            if model1_Cy_fw is not None:
                if verbose:
                    logger.info(
                        f'Initializing model1_Cy_fw with method "{init_method}"'
                    )
                model1_Cy_fw.model.set_weights(sInit.model1_Cy_fw.model.get_weights())
            if model1_FCy is not None:
                if verbose:
                    logger.info(f'Initializing model1_FCy with method "{init_method}"')
                model1_FCy.model.set_weights(sInit.model1_FCy.model.get_weights())
            if model1_FCy_fw is not None:
                if verbose:
                    logger.info(
                        f'Initializing model1_FCy_fw with method "{init_method}"'
                    )
                model1_FCy_fw.model.set_weights(sInit.model1_FCy_fw.model.get_weights())
            if model2 is not None:
                if verbose:
                    logger.info(f'Initializing RNN2 with method "{init_method}"')
                model2.set_cell_weights(sInit.model2.get_cell_weights())
                if epochs == 0:
                    model2.set_batch_size(1)
            if model2_Cz is not None:
                if verbose:
                    logger.info(f'Initializing model2_Cz with method "{init_method}"')
                model2_Cz.model.set_weights(sInit.model2_Cz.model.get_weights())
            if model2_Cz_fw is not None:
                if verbose:
                    logger.info(
                        f'Initializing model2_Cz_fw with method "{init_method}"'
                    )
                model2_Cz_fw.model.set_weights(sInit.model2_Cz_fw.model.get_weights())
        else:
            sInit = None

        if steps_ahead is not None:
            log = f"Steps ahead to use in loss: {steps_ahead}"
            if steps_ahead_loss_weights is not None:
                log += f" (weights: {steps_ahead_loss_weights})"
            logger.info(log)

        (
            steps_ahead,
            steps_ahead_loss_weights,
            steps_ahead_model1,
            steps_ahead_loss_weights_model1,
            model1_orig_step_inds,
        ) = self.get_model_steps_ahead(steps_ahead, steps_ahead_loss_weights)

        steps_aheadBU = copy.deepcopy(steps_ahead)
        if steps_ahead is None:
            steps_ahead = [1 if not self.bidirectional else 0]

        if len([s for s in steps_ahead if s not in [0, 1]]) > 0:
            raise (Exception(STEPS_AHEAD_MSG))

        get_steps_ahead_from_model1 = lambda L: (
            [item for ind, item in enumerate(L) if ind in model1_orig_step_inds]
            if L is not None
            else None
        )
        # np.testing.assert_equal(steps_ahead, get_steps_ahead_from_model1(steps_ahead_model1))

        need_fw_reg_models = np.any(np.array(steps_ahead) != 1) and (
            self.nu > 0 or self.has_Dyz
        )

        """
        if true_model is not None:
            from .evaluation import evaluateDecoding
            trueModelZp, trueModelYp, trueModelXp = true_model.predict(Y.T)
            perf = evaluateDecoding(Z.T, trueModelZp, Y.T, trueModelYp)
            if Y_validation is not None:
                trueModelZp_val, trueModelYp_val, trueModelXp_val = true_model.predict(Y_validation.T)
                perf_val = evaluateDecoding(Z_validation.T, trueModelZp_val, Y_validation.T, trueModelYp_val)
            for pm in ['meanCC', 'meanyCC']:
                print('Ideal training {}: {:.2g}'.format(pm, perf[pm]))
                if Y_validation is not None:
                    print('Ideal validation {}: {:.2g}'.format(pm, perf_val[pm]))
        """

        logs = {}

        rnn_fit_args = {
            "epochs": epochs,
            "init_attempts": init_attempts,
            "max_attempts": max_attempts,
            "throw_on_fail": throw_on_fail,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_measure": early_stopping_measure,
            "start_from_epoch": start_from_epoch_rnn,
        }
        reg_fit_args = {
            "epochs": epochs,
            "init_attempts": init_attempts,
            "max_attempts": max_attempts,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_measure": early_stopping_measure,
            "start_from_epoch": start_from_epoch_reg,
        }

        # Stage 1
        allXp1_steps, allXp1_steps_val = None, None
        allXp1U_steps, allXp1U_steps_val = None, None
        allZp1_steps, allZp1_steps_val = None, None
        if n1 > 0:
            if (
                epochs > 0
                or not skip_predictions
                or ((n2 > 0 or not skip_Cy) and epochs > 0)
            ):
                # Prepare input to RNN model1
                YU, FT_in = self.prepare_inputs_to_model1(Y, U)
                YU_validation, FT_in_validation = self.prepare_inputs_to_model1(
                    Y_validation, U_validation
                )
            if epochs > 0:
                # Learn RNN model1
                if verbose:
                    logger.info(f"Stage 1: Learning A11, K1, Cz1 (ZType: {ZType})")
                tic = time.perf_counter()
                history1 = model1.fit(
                    Y_in=YU,
                    Y_out=Z,
                    FT_in=FT_in,
                    Y_in_val=YU_validation,
                    Y_out_val=Z_validation,
                    FT_in_val=FT_in_validation,
                    **rnn_fit_args,
                )
                logs["model1"] = convertHistoryToDict(history1, tic=tic)
            """
            from .plot import plotPredictionScatter
            plotPredictionScatter(Z[:, isOkZ], allZp[:, isOkZ],
                    plot45DegLine=True, plotLSLine=True, styles={'size': 10, 'marker': 'x'},
                    addPerfMeasuresToTitle=['CC', 'R2'])
            """
            # Collect predictions for use in next steps
            if (
                not skip_predictions
                or ((n2 > 0 or not skip_Cy) and epochs > 0)
                or verbose
            ):
                preds1 = model1.predict(YU, FT_in=FT_in)
                allXp1_steps = preds1[: len(steps_ahead_model1)]
                allZp1_steps = preds1[
                    len(steps_ahead_model1) : 2 * len(steps_ahead_model1)
                ]
                allZp = allZp1_steps[0]

                if verbose:
                    logger.info(
                        "Training Z pred after fitting RNN1: \n"
                        + getLossLogStr(
                            ZTrue,
                            shift_ms_to_1s_series(
                                allZp1_steps,
                                steps_ahead_model1,
                                self.missing_marker,
                                time_first=False,
                            ),
                            steps_ahead_model1,
                            ZType,
                            ZLossFuncs,
                        )
                    )

                if Y_validation is not None:
                    preds1_val = model1.predict(YU_validation, FT_in=FT_in_validation)
                    allXp1_steps_val = preds1_val[: len(steps_ahead_model1)]
                    allZp1_steps_val = preds1_val[
                        len(steps_ahead_model1) : 2 * len(steps_ahead_model1)
                    ]
                    if verbose:
                        logger.info(
                            "Validation Z pred after fitting RNN1: \n"
                            + getLossLogStr(
                                Z_validation.T,
                                shift_ms_to_1s_series(
                                    allZp1_steps_val,
                                    steps_ahead_model1,
                                    self.missing_marker,
                                    time_first=False,
                                ),
                                steps_ahead_model1,
                                ZType,
                                ZLossFuncs,
                            )
                        )

                # Prepare input to regression model1_Cy (regression from state to neural activity)
                (
                    allXp1U_steps,
                    allXp1U_steps_Shifted,
                    YRep1,
                ) = self.prepare_inputs_to_model1_Cy(
                    Y, U, allXp1_steps, steps_ahead_model1
                )
                (
                    allXp1U_steps_val,
                    allXp1U_steps_Shifted_val,
                    YRep1_val,
                ) = self.prepare_inputs_to_model1_Cy(
                    Y_validation, U_validation, allXp1_steps_val, steps_ahead_model1
                )
                if (
                    need_fw_reg_models
                ):  # We have feedthrough and multi-step ahead, so we also need fw models
                    # Prepare input to regression model1_Cy_fw (regression from state to neural activity, without inputs)
                    allXp1_steps_Shifted = self.prepare_inputs_to_model1_Cy_fw(
                        allXp1_steps, steps_ahead_model1
                    )
                    allXp1_steps_Shifted_val = self.prepare_inputs_to_model1_Cy_fw(
                        allXp1_steps_val, steps_ahead_model1
                    )

            # Step 2
            if (n2 > 0 or skip_Cy is False) and not model1_Cy_Full:
                # Learn model1_Cy (regression from state to neural activity)
                (
                    steps_ahead_model1_Cy,
                    keeper_func_model1_Cy,
                ) = self.keep_steps_and_cat_for_reg_model(
                    get_steps_ahead_from_model1(steps_ahead_model1),
                    return_keeper_func=True,
                )
                if need_fw_reg_models:
                    (
                        steps_ahead_model1_Cy_fw,
                        keeper_func_model1_Cy_fw,
                    ) = self.keep_steps_and_cat_for_reg_model(
                        get_steps_ahead_from_model1(steps_ahead_model1),
                        is_fw=True,
                        return_keeper_func=True,
                    )
                if epochs > 0:
                    # Form neural regression training data considering required steps ahead
                    allXp1U_steps_shiftedCat = self.keep_steps_and_cat_for_reg_model(
                        get_steps_ahead_from_model1(allXp1U_steps_Shifted)
                    )
                    YRep1Cat = self.keep_steps_and_cat_for_reg_model(
                        get_steps_ahead_from_model1(YRep1)
                    )
                    allXp1U_val_steps_shiftedCat = (
                        self.keep_steps_and_cat_for_reg_model(
                            get_steps_ahead_from_model1(allXp1U_steps_Shifted_val)
                        )
                    )
                    Y_validation_Rep1Cat = self.keep_steps_and_cat_for_reg_model(
                        get_steps_ahead_from_model1(YRep1_val)
                    )

                    if verbose:
                        logger.info(f"Stage 1: Learning Cy1 (YType: {YType})")
                    tic = time.perf_counter()
                    history1_Cy = model1_Cy.fit(
                        allXp1U_steps_shiftedCat,
                        YRep1Cat,
                        X_in_val=allXp1U_val_steps_shiftedCat,
                        X_out_val=Y_validation_Rep1Cat,
                        **reg_fit_args,
                    )
                    logs["model1_Cy"] = convertHistoryToDict(history1_Cy, tic=tic)

                    if (
                        need_fw_reg_models
                    ):  # We have feedthrough and multi-step ahead, so we also need fw models (that wouldn't use the feedthrough)
                        # Form neural regression training data considering required steps ahead
                        allXp1_steps_shifted_fwCat = (
                            self.keep_steps_and_cat_for_reg_model(
                                get_steps_ahead_from_model1(allXp1_steps_Shifted),
                                is_fw=True,
                            )
                        )
                        YRep1_fwCat = self.keep_steps_and_cat_for_reg_model(
                            get_steps_ahead_from_model1(YRep1), is_fw=True
                        )
                        allXp1_val_steps_shifted_fwCat = (
                            self.keep_steps_and_cat_for_reg_model(
                                get_steps_ahead_from_model1(allXp1_steps_Shifted_val),
                                is_fw=True,
                            )
                        )
                        Y_validation_Rep1_fwCat = self.keep_steps_and_cat_for_reg_model(
                            get_steps_ahead_from_model1(YRep1_val), is_fw=True
                        )

                        if verbose:
                            logger.info(f"Stage 1: Learning Cy1_fw (YType: {YType})")
                        tic = time.perf_counter()
                        history1_Cy_fw = model1_Cy_fw.fit(
                            allXp1_steps_shifted_fwCat,
                            YRep1_fwCat,
                            X_in_val=allXp1_val_steps_shifted_fwCat,
                            X_out_val=Y_validation_Rep1_fwCat,
                            **reg_fit_args,
                        )
                        logs["model1_Cy_fw"] = convertHistoryToDict(
                            history1_Cy_fw, tic=tic
                        )

                if verbose or epochs > 0 or not skip_predictions:
                    allYp1_steps_Shifted = [
                        model1_Cy.predict(tmp) for tmp in allXp1U_steps_Shifted
                    ]  # n-step
                    allYp = allYp1_steps_Shifted[0]  # 1-step
                    logger.info(
                        "Training Y pred after fitting model1_Cy: \n"
                        + getLossLogStr(
                            YTrue,
                            keeper_func_model1_Cy(
                                get_steps_ahead_from_model1(allYp1_steps_Shifted)
                            ),
                            steps_ahead_model1_Cy,
                            YType,
                            YLossFuncs,
                        )
                    )

                    if (
                        need_fw_reg_models
                    ):  # We have feedthrough and multi-step ahead, so we also need fw models
                        allYp1_steps_Shifted_fw = [
                            model1_Cy_fw.predict(tmp) for tmp in allXp1_steps_Shifted
                        ]  # n-step
                        logger.info(
                            "Training Y pred after fitting model1_Cy_fw: \n"
                            + getLossLogStr(
                                YTrue,
                                keeper_func_model1_Cy_fw(
                                    get_steps_ahead_from_model1(allYp1_steps_Shifted)
                                ),
                                steps_ahead_model1_Cy_fw,
                                YType,
                                YLossFuncs,
                            )
                        )

            """
            # Plot training & validation loss values
            plt.plot(np.log10(history1.history['loss']), label='Train z-loss')
            plt.plot(np.log10(history1.history['val_loss']), label='Test z-loss')
            if true_z_mse is not None:
                plt.hlines(np.log10(true_z_mse), history1.epoch[0], history1.epoch[-1], label='Train (ideal)', linestyles='dashed')
            if true_z_mse_val is not None:
                plt.hlines(np.log10(true_z_mse_val), history1.epoch[0], history1.epoch[-1], label='Test (ideal)')
            plt.title('Model z-loss')
            plt.ylabel('log10(Loss)')
            plt.xlabel('Epoch')
            plt.legend(loc='upper left')
            plt.show()
            """

        if (epochs > 0 or not skip_predictions) and (n2 > 0 or skip_Cy is False):
            # Prepare input to RNN model2 and prior for model2_Cz
            # Find the residual neural activity
            Y_in_resU, allYp1_steps, n1_in = self.prepare_inputs_to_model2(
                model1_Cy, Y, U, allXp1_steps, allXp1U_steps
            )
            Y_in_res_valU, allYp1_steps_val, n1_in_val = self.prepare_inputs_to_model2(
                model1_Cy,
                Y_validation,
                U_validation,
                allXp1_steps_val,
                allXp1U_steps_val,
            )
            prior_preds = get_steps_ahead_from_model1(allYp1_steps)
            prior_preds_vals = get_steps_ahead_from_model1(allYp1_steps_val)

        # Stage 2
        allXp2_steps, allXp2_steps_val = None, None
        if n2 > 0:
            if epochs > 0:
                # Fit RNN model2
                if verbose:
                    logger.info(f"Stage 2: Learning A12, A22, K2, Cy2 (YType: {YType})")
                tic = time.perf_counter()
                history2 = model2.fit(
                    Y_in=Y_in_resU,
                    Y_out=Y,
                    FT_in=U,
                    n1_in=n1_in,
                    prior_pred=prior_preds,
                    Y_in_val=Y_in_res_valU,
                    Y_out_val=Y_validation,
                    FT_in_val=U_validation,
                    n1_in_val=n1_in_val,
                    prior_pred_val=prior_preds_vals,
                    prior_pred_shift_by_one=True,
                    **rnn_fit_args,
                )
                logs["model2"] = convertHistoryToDict(history2, tic=tic)
            if not skip_predictions or verbose:
                preds2 = model2.predict(
                    Y_in_resU,
                    FT_in=U,
                    n1_in=n1_in,
                    prior_pred=prior_preds,
                    prior_pred_shift_by_one=True,
                )
                allXp2_steps = preds2[: len(steps_ahead)]
                allYp = preds2[len(steps_ahead)]
                if Y_validation is not None:
                    preds2_val = model2.predict(
                        Y_in_res_valU,
                        FT_in=U_validation,
                        n1_in=n1_in_val,
                        prior_pred=prior_preds_vals,
                        prior_pred_shift_by_one=True,
                    )
                    allXp2_steps_val = preds2_val[: len(steps_ahead)]
                if verbose:
                    logger.info(
                        "Training Y pred after fitting RNN2: \n"
                        + getLossLogStr(
                            YTrue,
                            shift_ms_to_1s_series(
                                preds2[len(steps_ahead) : 2 * len(steps_ahead)],
                                steps_ahead,
                                self.missing_marker,
                                time_first=False,
                            ),
                            steps_ahead,
                            YType,
                            YLossFuncs,
                        )
                    )
                    if Y_validation is not None:
                        logger.info(
                            "Validation Y pred after fitting RNN2: \n"
                            + getLossLogStr(
                                Y_validation.T,
                                shift_ms_to_1s_series(
                                    preds2_val[len(steps_ahead) : 2 * len(steps_ahead)],
                                    steps_ahead,
                                    self.missing_marker,
                                    time_first=False,
                                ),
                                steps_ahead,
                                YType,
                                YLossFuncs,
                            )
                        )

            """
            # Plot training & validation loss values
            plt.plot(np.log10(history2.history['loss']), label='Train')
            plt.plot(np.log10(history2.history['val_loss']), label='Test')
            if true_mse is not None:
                plt.hlines(np.log10(true_y_mse), history2.epoch[0], history2.epoch[-1], label='Train (ideal)')
            if true_mse_val is not None:
                plt.hlines(np.log10(true_y_mse_val), history2.epoch[0], history2.epoch[-1], label='Test (ideal)')
            plt.title('Model loss')
            plt.ylabel('log10(Loss)')
            plt.xlabel('Epoch')
            plt.legend(loc='upper left')
            plt.show()
            """

        if not skip_predictions:
            allX_steps_val = None
            if n2 == 0:
                allX_steps = [
                    allXp1_steps[model1_orig_step_inds[saInd]]
                    for saInd in range(len(steps_ahead))
                ]
                allX_steps_val = (
                    [
                        allXp1_steps_val[model1_orig_step_inds[saInd]]
                        for saInd in range(len(steps_ahead))
                    ]
                    if allXp1_steps_val is not None
                    else None
                )
            elif n1 == 0:
                allX_steps = allXp2_steps
                if Y_validation is not None:
                    allX_steps_val = allXp2_steps_val
            else:
                allX_steps = [None for _ in range(len(steps_ahead))]
                allX_steps_val = [None for _ in range(len(steps_ahead))]
                for saInd in range(len(steps_ahead)):
                    allX_steps[saInd] = np.concatenate(
                        (
                            allXp1_steps[model1_orig_step_inds[saInd]],
                            allXp2_steps[saInd],
                        ),
                        axis=0,
                    )
                    if Y_validation is not None:
                        allX_steps_val[saInd] = np.concatenate(
                            (
                                allXp1_steps_val[model1_orig_step_inds[saInd]],
                                allXp2_steps_val[saInd],
                            ),
                            axis=0,
                        )
            allX = allX_steps[0]
        else:
            allX = np.empty((nx, Ndat))
            allX[:] = np.nan

        if (
            Z is not None
            and nz > 0
            and (model2_Cz_Full or n1 == 0 or (n1 > 0 and n2 > 0 and allow_nonzero_Cz2))
        ):
            if epochs > 0 or not skip_predictions:
                # Prepare input to regression model2_Cz (regression from stage 2 or all states to behavior)
                allXpForCz, priorForCzFit, ZRep2 = self.prepare_inputs_to_model2_Cz(
                    Y,
                    Z,
                    U,
                    allX_steps,
                    allXp2_steps,
                    get_steps_ahead_from_model1(allZp1_steps),
                    steps_ahead,
                )
                (
                    allXpForCz_val,
                    priorForCzFit_val,
                    ZRep2_val,
                ) = self.prepare_inputs_to_model2_Cz(
                    Y_validation,
                    Z_validation,
                    U_validation,
                    allX_steps_val,
                    allXp2_steps_val,
                    get_steps_ahead_from_model1(allZp1_steps_val),
                    steps_ahead,
                )

                steps_ahead_model2_Cz = self.keep_steps_and_cat_for_reg_model(
                    steps_ahead
                )
                allXpForCzCat = self.keep_steps_and_cat_for_reg_model(allXpForCz)
                priorForCzFitCat = self.keep_steps_and_cat_for_reg_model(priorForCzFit)
                ZRepCat = self.keep_steps_and_cat_for_reg_model(ZRep2)
                allXpForCz_valCat = self.keep_steps_and_cat_for_reg_model(
                    allXpForCz_val
                )
                priorForCzFit_valCat = self.keep_steps_and_cat_for_reg_model(
                    priorForCzFit_val
                )
                ZRep2_valCat = self.keep_steps_and_cat_for_reg_model(ZRep2_val)

                if (
                    need_fw_reg_models
                ):  # We have feedthrough and multi-step ahead, so we also need fw models
                    # Prepare input to regression model2_Cz_fw (regression from stage 2 or all states to behavior)
                    (
                        allXpForCz_fw,
                        priorForCzFit_fw,
                    ) = self.prepare_inputs_to_model2_Cz_fw(
                        allX_steps,
                        allXp2_steps,
                        get_steps_ahead_from_model1(allZp1_steps),
                        steps_ahead,
                    )
                    allXpForCz_fw_val, priorForCzFit_fw_val = (
                        self.prepare_inputs_to_model2_Cz_fw(
                            allX_steps_val,
                            allXp2_steps_val,
                            get_steps_ahead_from_model1(allZp1_steps_val),
                            steps_ahead,
                        )
                        if Y_validation is not None
                        else None
                    )
                    # Concatenate the relevant time steps
                    steps_ahead_model2_Cz_fw = self.keep_steps_and_cat_for_reg_model(
                        steps_ahead, is_fw=True
                    )
                    allXpForCz_fwCat = self.keep_steps_and_cat_for_reg_model(
                        allXpForCz_fw, is_fw=True
                    )
                    priorForCzFit_fwCat = self.keep_steps_and_cat_for_reg_model(
                        priorForCzFit_fw, is_fw=True
                    )
                    ZRep_fwCat = self.keep_steps_and_cat_for_reg_model(
                        ZRep2, is_fw=True
                    )
                    allXpForCz_fw_valCat = self.keep_steps_and_cat_for_reg_model(
                        allXpForCz_fw_val, is_fw=True
                    )
                    priorForCzFit_fw_valCat = self.keep_steps_and_cat_for_reg_model(
                        priorForCzFit_fw_val, is_fw=True
                    )
                    ZRep2_fw_valCat = self.keep_steps_and_cat_for_reg_model(
                        ZRep2_val, is_fw=True
                    )

            # Step 4
            # Learn Cz2 (regression from state to behavior)
            # if sInit is None and regression_init_method is not None:
            #     if verbose:
            #         logger.info('Initializing Cz with "{}" for attempt 1'.format(regression_init_method))
            #     from .LatentDecoder import ProjectionModel
            #     pm = ProjectionModel()
            #     pm.fit(allXpForCzCat, ZRepCat, U=None, method=regression_init_method, ZType=ZType, missing_marker=self.missing_marker)
            #     w = [pm.get_weights().T]
            #     model2_Cz.model.set_weights(w)

            #     if need_fw_reg_models: # We have feedthrough and multi-step ahead, so we also need fw models
            #         pm = ProjectionModel()
            #         pm.fit(allXpForCz_fwCat, ZRep_fwCat, U=None, method=regression_init_method, ZType=ZType, missing_marker=self.missing_marker)
            #         w = [pm.get_weights().T]
            #         model2_Cz_fw.model.set_weights(w)

            if epochs > 0:
                if verbose:
                    logger.info(f"Stage 2: Learning Cz2 (ZType: {ZType})")
                tic = time.perf_counter()
                history2_Cz = model2_Cz.fit(
                    allXpForCzCat,
                    ZRepCat,
                    prior_pred=priorForCzFitCat,
                    X_in_val=allXpForCz_valCat,
                    X_out_val=ZRep2_valCat,
                    prior_pred_val=priorForCzFit_valCat,
                    **reg_fit_args,
                )
                logs["model2_Cz"] = convertHistoryToDict(history2_Cz, tic=tic)

                if (
                    need_fw_reg_models
                ):  # We have feedthrough and multi-step ahead, so we also need fw models
                    if verbose:
                        logger.info(f"Stage 2: Learning Cz2_fw (ZType: {ZType})")
                    tic = time.perf_counter()
                    history2_Cz_fw = model2_Cz_fw.fit(
                        allXpForCz_fwCat,
                        ZRep_fwCat,
                        prior_pred=priorForCzFit_fwCat,
                        X_in_val=allXpForCz_fw_valCat,
                        X_out_val=ZRep2_fw_valCat,
                        prior_pred_val=priorForCzFit_fw_valCat,
                        **reg_fit_args,
                    )
                    logs["model2_Cz_fw"] = convertHistoryToDict(history2_Cz_fw, tic=tic)

            if not skip_predictions:
                allZp = model2_Cz.predict(
                    allXpForCz[0],
                    prior_pred=priorForCzFit[0] if priorForCzFit is not None else None,
                )
                if verbose:
                    allZp_steps_Shifted = [
                        model2_Cz.predict(
                            allXpForCz[saInd],
                            prior_pred=(
                                priorForCzFit[saInd]
                                if priorForCzFit is not None
                                else None
                            ),
                        )
                        for saInd in range(len(allXpForCz))
                    ]  # n-step
                    logger.info(
                        "Training Z pred after fitting model2_Cz: \n"
                        + getLossLogStr(
                            ZTrue,
                            allZp_steps_Shifted,
                            steps_ahead_model2_Cz,
                            ZType,
                            ZLossFuncs,
                        )
                    )

                    if (
                        need_fw_reg_models
                    ):  # We have feedthrough and multi-step ahead, so we also need fw models
                        allZp_steps_Shifted_fw = [
                            model2_Cz_fw.predict(
                                allXpForCz_fw[saInd],
                                prior_pred=(
                                    priorForCzFit_fw[saInd]
                                    if priorForCzFit_fw is not None
                                    else None
                                ),
                            )
                            for saInd in range(len(allXpForCz_fw))
                        ]  # n-step
                        logger.info(
                            "Training Z pred after fitting model2_Cz_fw: \n"
                            + getLossLogStr(
                                ZTrue,
                                allZp_steps_Shifted_fw,
                                steps_ahead_model2_Cz_fw,
                                ZType,
                                ZLossFuncs,
                            )
                        )

        if model1_Cy_Full:
            if epochs > 0 or not skip_predictions:
                # Prepare input to final regression model1_Cy (regression from all states to neural activity)
                (
                    allXpUForCy_steps,
                    allXpUForCy_steps_Shifted,
                    YRep2,
                ) = self.prepare_inputs_to_model1_Cy(Y, U, allX_steps, steps_ahead)
                (
                    allXpUForCy_steps_val,
                    allXpUForCy_steps_Shifted_val,
                    YRep2_val,
                ) = self.prepare_inputs_to_model1_Cy(
                    Y_validation, U_validation, allX_steps_val, steps_ahead
                )

                steps_ahead_model1_Cy = self.keep_steps_and_cat_for_reg_model(
                    steps_ahead
                )
                allXpUForCyCat = self.keep_steps_and_cat_for_reg_model(
                    allXpUForCy_steps_Shifted
                )
                YRep2Cat = self.keep_steps_and_cat_for_reg_model(YRep2)
                allXpUForCy_valCat = self.keep_steps_and_cat_for_reg_model(
                    allXpUForCy_steps_Shifted_val
                )
                Y_validation_Rep2Cat = self.keep_steps_and_cat_for_reg_model(YRep2_val)

                if (
                    need_fw_reg_models
                ):  # We have feedthrough and multi-step ahead, so we also need fw models
                    # Prepare input to final regression model1_Cy_fw (regression from all states to neural activity, without feedthrough for forward prediction)
                    allXpForCy_steps_Shifted = self.prepare_inputs_to_model1_Cy_fw(
                        allX_steps, steps_ahead
                    )
                    allXpForCy_steps_Shifted_val = self.prepare_inputs_to_model1_Cy_fw(
                        allX_steps_val, steps_ahead
                    )

                    steps_ahead_model1_Cy_fw = self.keep_steps_and_cat_for_reg_model(
                        steps_ahead, is_fw=True
                    )
                    allXpForCy_fwCat = self.keep_steps_and_cat_for_reg_model(
                        allXpForCy_steps_Shifted, is_fw=True
                    )
                    YRep2_fwCat = self.keep_steps_and_cat_for_reg_model(
                        YRep2, is_fw=True
                    )
                    allXpForCy_fw_valCat = self.keep_steps_and_cat_for_reg_model(
                        allXpForCy_steps_Shifted_val, is_fw=True
                    )
                    Y_validation_Rep2_fwCat = self.keep_steps_and_cat_for_reg_model(
                        YRep2_val, is_fw=True
                    )

            # Learn Cy (regression from all states to neural activity)
            if epochs > 0:
                if verbose:
                    logger.info(f"Learning full Cy (YType: {YType})")
                tic = time.perf_counter()
                history1_Cy = model1_FCy.fit(
                    allXpUForCyCat,
                    YRep2Cat,
                    X_in_val=allXpUForCy_valCat,
                    X_out_val=Y_validation_Rep2Cat,
                    **reg_fit_args,
                )
                logs["model1_FCy"] = convertHistoryToDict(history1_Cy, tic=tic)

                if (
                    need_fw_reg_models
                ):  # We have feedthrough and multi-step ahead, so we also need fw models
                    if verbose:
                        logger.info(f"Learning full Cy_fw (YType: {YType})")
                    tic = time.perf_counter()
                    history1_Cy_fw = model1_FCy_fw.fit(
                        allXpForCy_fwCat,
                        YRep2_fwCat,
                        X_in_val=allXpForCy_fw_valCat,
                        X_out_val=Y_validation_Rep2_fwCat,
                        **reg_fit_args,
                    )
                    logs["model1_FCy_fw"] = convertHistoryToDict(
                        history1_Cy_fw, tic=tic
                    )

            if not skip_predictions:
                allYp = model1_FCy.predict(allXpUForCy_steps[0])
                if verbose:
                    allYp_steps_Shifted = [
                        model1_FCy.predict(tmp) for tmp in allXpUForCy_steps_Shifted
                    ]  # n-step
                    logger.info(
                        "Training Y pred after fitting model1_FCy: \n"
                        + getLossLogStr(
                            YTrue, allYp_steps_Shifted, steps_ahead, YType, YLossFuncs
                        )
                    )
                    if (
                        need_fw_reg_models
                    ):  # We have feedthrough and multi-step ahead, so we also need fw models
                        allYp_steps_Shifted = [
                            model1_FCy_fw.predict(tmp)
                            for tmp in allXpForCy_steps_Shifted
                        ]  # n-step
                        logger.info(
                            "Training Y pred after fitting model1_FCy_fw: \n"
                            + getLossLogStr(
                                YTrue,
                                allYp_steps_Shifted,
                                steps_ahead,
                                YType,
                                YLossFuncs,
                            )
                        )

        if ny > 0 and YType == "cont" and not skip_Cy:
            if not skip_predictions and allYp is not None:
                YErr = allYp - Y
                YErrCov = np.cov(YErr.T, rowvar=False)
                YErrMean = np.mean(YErr.T, axis=0)
            if epochs == 0 and sInit is not None:
                YErrCov = sInit.YErrCov
                YErrMean = None
        else:
            YErrCov = None
            YErrMean = None

        if nz > 0 and not skip_predictions and allZp is not None and ZType == "cont":
            Z_res = allZp - Z
            ZErrCov = np.cov(Z_res.T, rowvar=False)
            ZErrMean = np.mean(Z_res.T, axis=0)
        else:
            ZErrCov = None
            ZErrMean = None

        self.YPrepMap = YPrepMap
        self.ZPrepMap = ZPrepMap
        self.UPrepMap = UPrepMap

        self.model1 = model1
        self.model1_Cy = model1_Cy
        self.model1_Cy_fw = model1_Cy_fw
        self.model1_FCy = model1_FCy
        self.model1_FCy_fw = model1_FCy_fw
        self.model2 = model2
        self.model2_Cz = model2_Cz
        self.model2_Cz_fw = model2_Cz_fw

        self.logs = logs if save_logs else {}

        # Residual stats
        self.YErrCov = YErrCov
        self.YErrMean = YErrMean
        self.ZErrCov = ZErrCov
        self.ZErrMean = ZErrMean

        # [Additional stats]
        self.XCov = np.cov(allX, rowvar=True)
        if np.any(np.isnan(self.XCov)):
            logger.warning(
                f"Some states have blown up leading to bad XCov: {self.XCov}"
            )
            if throw_on_fail:
                raise (Exception(f"Learned model was not stable!"))
        self.YCov = np.cov(Y, rowvar=True)
        if Z is not None:
            self.ZCov = np.cov(Z, rowvar=True)
        set_global_tf_eagerly_flag(eagerly_flag_backup)

    def keep_steps_and_cat_for_reg_model(
        self, input_list_shifted, is_fw=False, return_keeper_func=False
    ):
        (
            steps_ahead,
            steps_ahead_loss_weights,
            steps_ahead_model1,
            steps_ahead_loss_weights_model1,
            model1_orig_step_inds,
        ) = self.get_model_steps_ahead(self.steps_ahead, self.steps_ahead_loss_weights)

        if is_fw:
            # Is intended for _fw regression models, so only keeping weighted steps_ahead that are more than 1
            weighted_steps_ahead_inds = [
                saInd
                for saInd in range(len(steps_ahead))
                if (
                    steps_ahead_loss_weights is None
                    or steps_ahead_loss_weights[saInd] != 0
                )
                and steps_ahead[saInd] != 1
            ]
        else:
            if (
                self.nu == 0
            ):  # No feedthrough, so can keep all steps ahead for non-fw models
                weighted_steps_ahead_inds = [
                    saInd
                    for saInd in range(len(steps_ahead))
                    if (
                        steps_ahead_loss_weights is None
                        or steps_ahead_loss_weights[saInd] != 0
                    )
                ]
            else:  # we have input and thus feedthrough so will only keep 1-step ahead for regression models
                weighted_steps_ahead_inds = [
                    saInd
                    for saInd in range(len(steps_ahead))
                    if (
                        steps_ahead_loss_weights is None
                        or steps_ahead_loss_weights[saInd] != 0
                    )
                    and steps_ahead[saInd] == 1
                ]

        if (
            steps_ahead_loss_weights is not None
            and len(
                np.unique(
                    [
                        w
                        for ind, w in enumerate(steps_ahead_loss_weights)
                        if ind in weighted_steps_ahead_inds
                    ]
                )
            )
            > 1
        ):
            raise (
                Exception(
                    "Fitting model1_Cy and model2_Cz with different step ahead weights is not supported yet!"
                )
            )

        def keep_w_items(x):
            # Keeps only elements of the list for which the corresponding steps_ahead loss weight is not zero
            if x is None:
                return None
            else:
                try:
                    out = [x[ind] for ind in weighted_steps_ahead_inds]
                except Exception as e:
                    print("WHAT?!")
                return out if isinstance(out, (tuple, list)) else [out]

        if input_list_shifted is not None:
            kept_items = keep_w_items(input_list_shifted)
            if isinstance(kept_items[0], (np.ndarray)):
                missing_inds_at_start = [
                    sav - 1 if sav >= 1 else 0 for sav in keep_w_items(steps_ahead)
                ]  # This many samples of predictions will be nan's and should be removed
                kept_items_ok = [
                    kept_items[i][:, missing_inds_at_start[i] :]
                    for i in range(len(kept_items))
                ]
                out = np.concatenate(kept_items_ok, axis=1)
            else:
                out = kept_items
        else:
            out = None
        if not return_keeper_func:
            return out
        else:
            return out, keep_w_items

    def prepare_inputs_to_model1_Cy(self, Y, U, allXp1_steps, steps_ahead):
        """Returns shifted inputs and repeated outputs needed to train model1_Cy

        Args:
            Y (_type_): _description_
            U (_type_): _description_
            allXp1_steps (_type_): _description_
            steps_ahead (_type_): _description_

        Returns:
            _type_: _description_
        """
        if Y is None:
            return None, None, None
        if self.nu > 0:
            raise (Exception(INPUT_MSG))
        else:
            allXp1U_steps = allXp1_steps

        allXp1U_steps_Shifted = shift_ms_to_1s_series(
            allXp1U_steps, steps_ahead, self.missing_marker, time_first=False
        )
        YRep = [Y for _ in range(len(steps_ahead))]
        return allXp1U_steps, allXp1U_steps_Shifted, YRep

    def prepare_inputs_to_model1_Cy_fw(self, allXp1_steps, steps_ahead):
        """Returns shifted inputs and repeated outputs needed to train model1_Cy

        Args:
            allXp1_steps (_type_): _description_
            steps_ahead (_type_): _description_

        Returns:
            _type_: _description_
        """
        allXp1_steps_Shifted = shift_ms_to_1s_series(
            allXp1_steps, steps_ahead, self.missing_marker, time_first=False
        )
        return allXp1_steps_Shifted

    def prepare_inputs_to_model2_Cz(
        self, Y, Z, U, allX_steps, allXp2_steps, allZp_steps, steps_ahead
    ):
        """Returns shifted inputs, shifted priors, and repeated outputs required to train model2_Cz

        Args:
            Y (_type_): _description_
            Z (_type_): _description_
            U (_type_): _description_
            allX_steps (_type_): _description_
            allXp2_steps (_type_): _description_
            allZp_steps (_type_): _description_
            steps_ahead (_type_): _description_

        Returns:
            _type_: _description_
        """
        if Y is None:
            return None, None, None

        if not self.model2_Cz_Full:
            allXpForCz = shift_ms_to_1s_series(
                allXp2_steps, steps_ahead, self.missing_marker, time_first=False
            )
            priorForCzFit = (
                shift_ms_to_1s_series(
                    allZp_steps, steps_ahead, self.missing_marker, time_first=False
                )
                if allZp_steps is not None
                else None
            )
        else:
            allXpForCz = shift_ms_to_1s_series(
                allX_steps, steps_ahead, self.missing_marker, time_first=False
            )
            priorForCzFit = None
        if self.has_Dyz and (
            self.n1 == 0 or self.model2_Cz_Full
        ):  # If Dyz not included in model1, then we need to learn it in model2_Cz
            allXpForCz = [np.concatenate((tmp_step, Y)) for tmp_step in allXpForCz]
        if self.nu > 0:
            raise (Exception(INPUT_MSG))

        ZRep = [Z for _ in range(len(steps_ahead))]
        return allXpForCz, priorForCzFit, ZRep

    def prepare_inputs_to_model2_Cz_fw(
        self, allX_steps, allXp2_steps, allZp_steps, steps_ahead
    ):
        """Returns shifted inputs, shifted priors, and repeated outputs required to train model2_Cz

        Args:
            allX_steps (_type_): _description_
            allXp2_steps (_type_): _description_
            allZp_steps (_type_): _description_
            steps_ahead (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not self.model2_Cz_Full:
            allXpForCz_fw = shift_ms_to_1s_series(
                allXp2_steps, steps_ahead, self.missing_marker, time_first=False
            )
            priorForCzFit_fw = (
                shift_ms_to_1s_series(
                    allZp_steps, steps_ahead, self.missing_marker, time_first=False
                )
                if allZp_steps is not None
                else None
            )
        else:
            allXpForCz_fw = shift_ms_to_1s_series(
                allX_steps, steps_ahead, self.missing_marker, time_first=False
            )
            priorForCzFit_fw = None

        return allXpForCz_fw, priorForCzFit_fw

    def prepare_inputs_to_model1(self, Y, U):
        """Returns concatenated Y and U inputs required to train RNN model1

        Args:
            Y (_type_): _description_
            U (_type_): _description_

        Returns:
            _type_: _description_
        """
        if Y is None:
            return None, None
        if self.nu > 0:
            raise (Exception(INPUT_MSG))
        else:
            YU = Y
        if self.has_Dyz:
            FT_in = YU
        else:
            FT_in = U
        return YU, FT_in

    def prepare_inputs_to_model2(self, model1_Cy, Y, U, allXp1_steps, allXp1U_steps):
        """Returns concatenated Y, U, X's from stage 1, forward prediction inputs n1_in, and prior predictions required to train RNN model2

        Args:
            model1_Cy (_type_): _description_
            Y (_type_): _description_
            U (_type_): _description_
            allXp1_steps (_type_): _description_
            allXp1U_steps (_type_): _description_

        Returns:
            _type_: _description_
        """
        if Y is None:
            return None, None, None
        if self.n1 > 0:
            allXp = allXp1_steps[0]  # 1-step (or 0-step for bidirectional)
            Y_in_res = np.concatenate((allXp, Y), axis=0)
        else:
            Y_in_res = Y
        if self.nu > 0:
            raise (Exception(INPUT_MSG))
        else:
            Y_in_resU = Y_in_res

        # Compute forward pred inputs for model 2
        n1_in = allXp1_steps if self.n1 > 0 else None

        # Compute prior for model 2
        if self.n1 > 0:
            allYp_steps = [
                model1_Cy.predict(allXp1U_this_step)
                for allXp1U_this_step in allXp1U_steps
            ]  # With original delays
        else:
            allYp_steps = None
        return Y_in_resU, allYp_steps, n1_in

    def build_models(self):
        _, _, yDist = self.prep_observation_for_training(None, self.YType)
        _, _, zDist = self.prep_observation_for_training(None, self.ZType)

        (
            steps_ahead,
            steps_ahead_loss_weights,
            steps_ahead_model1,
            steps_ahead_loss_weights_model1,
            model1_orig_step_inds,
        ) = self.get_model_steps_ahead(self.steps_ahead, self.steps_ahead_loss_weights)

        need_fw_reg_models = np.any(np.array(steps_ahead) != 1) and (
            self.nu > 0 or self.has_Dyz
        )

        if steps_ahead != [0] and steps_ahead != [1] and self.bidirectional:
            raise (
                Exception(f"Multistep ahead with bidirectional RNN is not supported!")
            )
        if steps_ahead == [1] and self.bidirectional:
            logger.info(
                f"Bidirectional must be used with steps_ahead=0, but steps_ahead=1. Allowing now to enable loading of some old results, but don't do this!"
            )

        # Step 1
        if self.n1 > 0:
            nft = self.ny + self.nu if self.has_Dyz else self.nu
            rnn_cell_args = copy.deepcopy(
                {
                    "ASettings": self.A1_args,
                    "KSettings": self.K1_args,
                    "CSettings": self.Cz1_args,
                    "learn_initial_state": self.learn_initial_state,
                }
            )
            this_log_dir = (
                "" if self.log_dir == "" else os.path.join(self.log_dir, "RNN1")
            )
            # model1: RNN with input: [y,u], states: x1 => n1, outputs: z, feedthrough: u
            model1 = RNNModel(
                self.n1,
                self.ny + self.nu,
                self.block_samples,
                self.batch_size,
                ny_out=self.nz,
                nft=nft,
                linear_cell=self.linear_cell,
                LSTM_cell=self.LSTM_cell,
                bidirectional=self.bidirectional,
                stateful=self.stateful,
                out_dist=zDist,
                optimizer_name=self.optimizer_name,
                optimizer_args=self.optimizer_args,
                lr_scheduler_name=self.lr_scheduler_name,
                lr_scheduler_args=self.lr_scheduler_args,
                enable_forward_pred=self.enable_forward_pred,
                steps_ahead=steps_ahead_model1,
                steps_ahead_loss_weights=steps_ahead_loss_weights_model1,
                name="RNN1_",
                log_dir=this_log_dir,
                missing_marker=self.missing_marker,
                cell_args=rnn_cell_args,
            )  # To predict z from past y
        else:
            model1 = None

        # Step 2
        if self.n2 > 0:
            rnn_cell_args = copy.deepcopy(
                {
                    "ASettings": self.A2_args,
                    "KSettings": self.K2_args,
                    "CSettings": self.Cy2_args,
                    "learn_initial_state": self.learn_initial_state,
                }
            )
            this_log_dir = (
                "" if self.log_dir == "" else os.path.join(self.log_dir, "RNN2")
            )
            # model2: RNN with input: [x1,y,u], states: x2 => n2, outputs: y, feedthrough: u
            n1_in = 2 * self.n1 if self.bidirectional else self.n1
            model2 = RNNModel(
                self.n2,
                n1_in + self.ny + self.nu,
                self.block_samples,
                self.batch_size,
                ny_out=self.ny,
                nft=self.nu,
                n1_in=n1_in,
                linear_cell=self.linear_cell,
                LSTM_cell=self.LSTM_cell,
                bidirectional=self.bidirectional,
                stateful=self.stateful,
                has_prior_pred=True,  # From stage 1
                out_dist=yDist,
                optimizer_name=self.optimizer_name,
                optimizer_args=self.optimizer_args,
                lr_scheduler_name=self.lr_scheduler_name,
                lr_scheduler_args=self.lr_scheduler_args,
                enable_forward_pred=self.enable_forward_pred,
                steps_ahead=steps_ahead,
                steps_ahead_loss_weights=steps_ahead_loss_weights,
                name="RNN2_",
                log_dir=this_log_dir,
                missing_marker=self.missing_marker,
                cell_args=rnn_cell_args,
            )  # To predict y from past y
        else:
            model2 = None

        # Step 4
        if self.nz > 0 and (
            self.model2_Cz_Full
            or self.n1 == 0
            or (self.n1 > 0 and self.n2 > 0 and self.allow_nonzero_Cz2)
        ):
            reg_args = copy.deepcopy(self.Cz2_args)
            reg_args["has_prior_pred"] = True  # From stage 1
            this_log_dir = (
                "" if self.log_dir == "" else os.path.join(self.log_dir, "Cz2")
            )
            # model2_Cz [if not full]: regression with input: [x2], outputs: z
            # model2_Cz [if full]: regression with input: [x1,x2,u], outputs: z
            input_dim = self.n2 if not self.model2_Cz_Full else self.n1 + self.n2
            if self.bidirectional:
                input_dim *= 2
            if self.has_Dyz and (self.n1 == 0 or self.model2_Cz_Full):
                input_dim += self.ny
            input_dim += self.nu
            model2_Cz = RegressionModel(
                input_dim,  # Without feedthrough if nu > 0 and with Dyz
                self.nz,
                name="model2_Cz_",
                log_dir=this_log_dir,
                missing_marker=self.missing_marker,
                optimizer_name=self.optimizer_name,
                optimizer_args=self.optimizer_args,
                lr_scheduler_name=self.lr_scheduler_name,
                lr_scheduler_args=self.lr_scheduler_args,
                **reg_args,
            )
            if need_fw_reg_models:
                this_log_dir = (
                    "" if self.log_dir == "" else os.path.join(self.log_dir, "Cz2_fw")
                )
                input_dim = self.n2 if not self.model2_Cz_Full else self.n1 + self.n2
                model2_Cz_fw = RegressionModel(
                    input_dim,  # Without feedthrough even if nu > 0 and without Dyz
                    self.nz,
                    name="model2_Cz_fw_",
                    log_dir=this_log_dir,
                    missing_marker=self.missing_marker,
                    optimizer_name=self.optimizer_name,
                    optimizer_args=self.optimizer_args,
                    lr_scheduler_name=self.lr_scheduler_name,
                    lr_scheduler_args=self.lr_scheduler_args,
                    **reg_args,
                )
            else:
                model2_Cz_fw = None
        else:
            model2_Cz, model2_Cz_fw = None, None

        # Step 2
        if self.n1 > 0 and (self.n2 > 0 or self.skip_Cy is False):
            reg_args = copy.deepcopy(self.Cy1_args)
            this_log_dir = (
                "" if self.log_dir == "" else os.path.join(self.log_dir, "Cy1")
            )
            # model1_Cy [if not full]: regression with input: [x1,u], outputs: y
            input_dim = self.n1 + self.nu
            if self.bidirectional:
                input_dim = 2 * self.n1 + self.nu
            model1_Cy = RegressionModel(
                input_dim,  # With feedthrough if nu > 0
                self.ny,
                name="model1_Cy_",
                log_dir=this_log_dir,
                missing_marker=self.missing_marker,
                optimizer_name=self.optimizer_name,
                optimizer_args=self.optimizer_args,
                lr_scheduler_name=self.lr_scheduler_name,
                lr_scheduler_args=self.lr_scheduler_args,
                **reg_args,
            )
            if need_fw_reg_models:
                this_log_dir = (
                    "" if self.log_dir == "" else os.path.join(self.log_dir, "Cy1_fw")
                )
                # model1_Cy_fw [if not full]: regression with input: [x1], outputs: y
                model1_Cy_fw = RegressionModel(
                    self.n1,  # Without feedthrough even if nu > 0
                    self.ny,
                    name="model1_Cy_fw_",
                    log_dir=this_log_dir,
                    missing_marker=self.missing_marker,
                    optimizer_name=self.optimizer_name,
                    optimizer_args=self.optimizer_args,
                    lr_scheduler_name=self.lr_scheduler_name,
                    lr_scheduler_args=self.lr_scheduler_args,
                    **reg_args,
                )
            else:
                model1_Cy_fw = None
        else:
            model1_Cy, model1_Cy_fw = None, None

        model1_FCy, model1_FCy_fw = None, None
        if self.model1_Cy_Full:
            reg_args = copy.deepcopy(self.Cy2_args)
            this_log_dir = (
                "" if self.log_dir == "" else os.path.join(self.log_dir, "Cy1")
            )
            # model1_Cy [if full]: regression with input: [x1,x2,u], outputs: y
            input_dim = self.n1 + self.n2 + self.nu  # With feedthrough if nu > 0
            if self.bidirectional:
                input_dim = 2 * (self.n1 + self.n2) + self.nu
            model1_FCy = RegressionModel(
                input_dim,
                self.ny,
                name="model1_FCy_",
                log_dir=this_log_dir,
                missing_marker=self.missing_marker,
                optimizer_name=self.optimizer_name,
                optimizer_args=self.optimizer_args,
                lr_scheduler_name=self.lr_scheduler_name,
                lr_scheduler_args=self.lr_scheduler_args,
                **reg_args,
            )
            if need_fw_reg_models:
                this_log_dir = (
                    "" if self.log_dir == "" else os.path.join(self.log_dir, "Cy1_fw")
                )
                # model1_Cy_fw [if full]: regression with input: [x1,x2], outputs: y
                model1_FCy_fw = RegressionModel(
                    self.n1 + self.n2,  # Without feedthrough even if nu > 0
                    self.ny,
                    name="model1_FCy_fw_",
                    log_dir=this_log_dir,
                    missing_marker=self.missing_marker,
                    optimizer_name=self.optimizer_name,
                    optimizer_args=self.optimizer_args,
                    lr_scheduler_name=self.lr_scheduler_name,
                    lr_scheduler_args=self.lr_scheduler_args,
                    **reg_args,
                )

        return (
            model1,
            model1_Cy,
            model1_Cy_fw,
            model2,
            model2_Cz,
            model2_Cz_fw,
            model1_FCy,
            model1_FCy_fw,
        )

    def print_summary(self):
        out = ""
        out = f"DPAD model summary (nx={self.nx}, ny={self.ny}, nz={self.nz}, nu={self.nu}, steps ahead: {self.steps_ahead}):"
        if self.model1 is not None:
            out += "\n- RNN model1: {}D state, {}D in, {}D out, {}D feedthrough, has prior: {}".format(
                self.model1.nx,
                self.model1.ny,
                self.model1.ny_out,
                self.model1.nft,
                self.model1.has_prior_pred,
            )
        if self.model2 is not None:
            out += "\n- RNN model2: {}D state, {}D in ({}D x1 + {}D y), {}D out, {}D feedthrough, has prior: {}".format(
                self.model2.nx,
                self.model2.ny,
                self.n1,
                self.ny,
                self.model2.ny_out,
                self.model2.nft,
                self.model2.has_prior_pred,
            )
        if self.model1_Cy is not None:
            out += "\n- Reg model1_Cy: {}D in, {}D out, full refit: {}".format(
                self.model1_Cy.n_in, self.model1_Cy.n_out, self.model1_Cy_Full
            )
        if self.model1_Cy_fw is not None:
            out += "\n- Reg model1_Cy_fw: {}D in, {}D out, full refit: {}".format(
                self.model1_Cy_fw.n_in, self.model1_Cy_fw.n_out, self.model1_Cy_Full
            )
        if self.model1_FCy is not None:
            out += "\n- Reg model1_FCy: {}D in, {}D out, full refit: {}".format(
                self.model1_FCy.n_in, self.model1_FCy.n_out, self.model1_Cy_Full
            )
        if self.model1_FCy_fw is not None:
            out += "\n- Reg model1_FCy_fw: {}D in, {}D out, full refit: {}".format(
                self.model1_FCy_fw.n_in, self.model1_FCy_fw.n_out, self.model1_Cy_Full
            )
        if self.model2_Cz is not None:
            out += "\n- Reg model2_Cz: {}D in, {}D out, full refit: {}".format(
                self.model2_Cz.n_in, self.model2_Cz.n_out, self.model2_Cz_Full
            )
        if self.model2_Cz_fw is not None:
            out += "\n- Reg model2_Cz_fw: {}D in, {}D out, full refit: {}".format(
                self.model2_Cz_fw.n_in, self.model2_Cz_fw.n_out, self.model2_Cz_Full
            )
        print(out)
        return out

    def plot_log_traces(
        self, ax, logs=None, label="", keys=None, linestyles=None, colors=None
    ):
        if logs is None:
            logs = self.logs
        if keys is None:
            keys = ["loss", "val_loss", "learning_rate"]
        if linestyles is None:
            lineStyles = ["-", "--", "-.", ":"]
        if colors is None:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
        for mi, (model_name, log) in enumerate(logs.items()):
            color = colors[mi % len(colors)]
            labelThis = f"{label}{model_name}"
            if "fit_time" in log and log["fit_time"] is not None:
                labelThis += " ({:.2g} mins) ".format(log["fit_time"] / 60)
            for ki, key in enumerate(keys):
                if key in log["history"]:
                    if (
                        key in ["learning_rate"]
                        and len(np.unique(log["history"][key])) == 1
                    ):
                        plt.text(
                            0.01,
                            0.99,
                            "{} = {:.2g}".format(key, log["history"][key][0]),
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                        )
                        continue
                    ax.plot(
                        log["epoch"],
                        np.log10(log["history"][key]),
                        c=color,
                        label=f"{labelThis}{key}",
                        linestyle=lineStyles[ki % len(lineStyles)],
                    )
                    if "picked_epoch" in log["params"]:
                        pick_ind = np.where(
                            np.array(log["epoch"]) == log["params"]["picked_epoch"]
                        )[0][0]
                    elif "best_epoch" in log["params"]:
                        pick_ind = log["params"]["best_epoch"]
                    elif "stopped_epoch" in log["params"]:
                        pick_ind = log["params"]["stopped_epoch"]
                    else:
                        pick_ind = None
                    if pick_ind is not None:
                        ax.scatter(
                            log["epoch"][pick_ind],
                            np.log10(log["history"][key][pick_ind]),
                            c=color,
                            marker="x",
                        )

    def plot_logs(
        self,
        logs=None,
        keys=None,
        titleStr="",
        skip_existing=False,
        saveFile=None,
        saveExtensions=None,
        fig=None,
        figsize=None,
    ):
        if skip_existing and checkIfAllExtsAlreadyExist(saveFile, saveExtensions):
            logger.info("Skipping... figure already exists: " + saveFile)
            return
        if logs is None:
            logs = self.logs
        if keys is None:
            keys = ["loss", "val_loss", "learning_rate"]
        if figsize is None:
            figsize = (6, 4)
        if fig is None:
            fig = plt.figure(figsize=figsize)
        else:
            plt.figure(fig)
        ax = fig.add_subplot(1, 1, 1)
        self.plot_log_traces(ax, logs=logs, keys=keys)
        ax.set_ylabel(f"log loss")
        ax.legend()
        ax.set_title(titleStr)
        showOrSaveFig(fig, saveFile, saveExtensions)
        return fig, ax

    def set_steps_ahead(self, steps_ahead, update_rnn_model_steps=True):
        steps_ahead, _, steps_ahead_model1, _, _ = self.get_model_steps_ahead(
            steps_ahead
        )
        self.steps_ahead = steps_ahead
        if self.model1 is not None and update_rnn_model_steps:
            self.model1.set_steps_ahead(steps_ahead_model1)
        if self.model2 is not None and update_rnn_model_steps:
            self.model2.set_steps_ahead(steps_ahead)

    def set_multi_step_with_A_KC(self, multi_step_with_A_KC):
        if self.model1 is not None:
            self.model1.set_multi_step_with_A_KC(multi_step_with_A_KC)
        if self.model2 is not None:
            self.model2.set_multi_step_with_A_KC(multi_step_with_A_KC)

    def set_multi_step_with_data_gen(
        self, multi_step_with_data_gen, update_rnn_model_steps=True, noise_samples=0
    ):
        self.multi_step_with_data_gen = multi_step_with_data_gen
        self.multi_step_data_gen_noise_samples = noise_samples
        if self.model1 is not None and update_rnn_model_steps:
            self.model1.set_steps_ahead([1])
        if self.model2 is not None and update_rnn_model_steps:
            self.model2.set_steps_ahead([1])

    def set_use_feedthrough_in_fw(self, use_feedthrough_in_fw):
        self.use_feedthrough_in_fw = use_feedthrough_in_fw
        if self.model1 is not None:
            self.model1.use_feedthrough_in_fw = use_feedthrough_in_fw
        if self.model2 is not None:
            self.model2.use_feedthrough_in_fw = use_feedthrough_in_fw

    def discardModels(self):
        """Prepares the object for pickling by replacing tf models with
        dictionaries of their weights
        """
        if (
            hasattr(self, "yPrepModel")
            and self.yPrepModel is not None
            and hasattr(self.yPrepModel, "discardModels")
        ):
            self.yPrepModel.discardModels()
        model_names = [
            "model1",
            "model2",
            "model1_Cy",
            "model1_Cy_fw",
            "model1_FCy",
            "model1_FCy_fw",
            "model2_Cz",
            "model2_Cz_fw",
        ]
        for model_name in model_names:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                if model is not None:
                    try:
                        setattr(self, model_name, model.get_recreation_info())
                    except:
                        setattr(self, model_name, model.model.get_weights())
        if hasattr(self, "initE") and hasattr(self.initE, "discardModels"):
            self.initE.discardModels()

    def restoreModels(self):
        """Prepares the object for use after loading from a pickled file
        by creating tf models and populating them with the saved weights
        """
        if (
            hasattr(self, "yPrepModel")
            and self.yPrepModel is not None
            and hasattr(self.yPrepModel, "restoreModels")
        ):
            self.yPrepModel.restoreModels()
        # For backwards compatibility with older results
        if not hasattr(self, "has_Dyz"):
            self.has_Dyz = False
        if not hasattr(self, "A1_args"):
            self.A1_args = self.A_args
        if not hasattr(self, "A2_args"):
            self.A2_args = self.A_args
        if not hasattr(self, "K1_args"):
            self.K1_args = self.K_args
        if not hasattr(self, "K2_args"):
            self.K2_args = self.K_args
        if not hasattr(self, "Cy1_args"):
            self.Cy1_args = self.Cy_args
        if not hasattr(self, "Cy2_args"):
            self.Cy2_args = self.Cy_args
        if not hasattr(self, "Cz1_args"):
            self.Cz1_args = self.Cz_args
        if not hasattr(self, "Cz2_args"):
            self.Cz2_args = self.Cz_args
        if not hasattr(self, "model1_Cy_fw"):
            self.model1_Cy_fw = None
        if not hasattr(self, "model2_Cz_fw"):
            self.model2_Cz_fw = None
        if not hasattr(self, "model1_FCy"):
            self.model1_FCy = None
        if not hasattr(self, "model1_FCy_fw"):
            self.model1_FCy_fw = None
        if not hasattr(self, "steps_ahead"):
            self.steps_ahead = None
        if not hasattr(self, "bidirectional"):
            self.bidirectional = False

        def fix_backward_compatibility_with_old_models(saved_model, new_model):
            if not self.bidirectional and len(saved_model.weights) != len(
                new_model.rnn.cell.get_weights()
            ):  # Unexpected, may be a backward compatibility issue with old models without Afw
                expected_weights = new_model.get_cell_weights()
                # [w.shape for w in new_model.rnn.get_weights()]
                # [w.shape for w in saved_model.weights]
                if "LSTMCell" in expected_weights:
                    if new_model.rnn.cell.unifiedAK and new_model.rnn.cell.unifiedAK:
                        print("To Do!")
                    else:
                        # New weight order: 'LSTMCell', 'K', 'C', 'LSTMCellfw', 'inital_state'
                        # Saved old weight order: 'LSTMCell', 'K', 'C', 'LSTMCellfw', 'inital_state'
                        fixed_weights = saved_model.weights[
                            : (
                                len(expected_weights["LSTMCell"])
                                + len(expected_weights["K"])
                                + len(expected_weights["C"])
                            )
                        ]  # The learned A (unified A+K) and C
                        fixed_weights += expected_weights[
                            "LSTMCellfw"
                        ]  # New parameter, not learned, not used, so keep as is in instantiated model
                        if "Kfw" in expected_weights:
                            fixed_weights += expected_weights[
                                "Kfw"
                            ]  # New parameter, not learned, not used, so keep as is in instantiated model
                        fixed_weights += saved_model.weights[-1:]  # Initial state
                elif (
                    new_model.rnn.cell.unifiedAK and new_model.rnn.cell.unifiedAK
                ):  # Let's fix this specific backward incompatible case for old models
                    fixed_weights = saved_model.weights[
                        : (len(expected_weights["A"]) + len(expected_weights["C"]))
                    ]  # The learned A (unified A+K) and C
                    fixed_weights += expected_weights[
                        "Afw"
                    ]  # New parameter, not learned, not used, so keep as is in instantiated model
                    if "Kfw" in expected_weights:
                        fixed_weights += expected_weights[
                            "Kfw"
                        ]  # New parameter, not learned, not used, so keep as is in instantiated model
                    fixed_weights += saved_model.weights[-1:]  # Initial state
                else:
                    fixed_weights = None
                if fixed_weights is not None:
                    # logger.warning(f'Fixed a backward incompatibility issue with old models learned with older code: \n- (Before) Loaded weights list ({len(saved_model.weights)} items): {saved_model.weights}\n- (After) Fixed weights list ({len(fixed_weights)} items): {fixed_weights})')
                    logger.warning(
                        f"Fixed a backward incompatibility issue with old models learned with older code: \n- (Before) Loaded weights list ({len(saved_model.weights)} items)\n- (After) Fixed weights list ({len(fixed_weights)} items)"
                    )
                    # [w.shape for w in fixed_weights]
                    saved_model.weights = fixed_weights
            return saved_model

        _, _, yDist = self.prep_observation_for_training(None, self.YType)
        _, _, zDist = self.prep_observation_for_training(None, self.ZType)
        if self.n1 > 0 and not isinstance(self.model1, (RNNModel)):
            if isinstance(self.model1, ReconstructionInfo) and hasattr(
                self.model1, "constructor_kwargs"
            ):
                if (
                    "smooth_input" in self.model1.constructor_kwargs
                    and not self.model1.constructor_kwargs["smooth_input"]
                ):  # For backward compatibility
                    self.model1.constructor_kwargs.pop("smooth_input")
                    self.model1.constructor_kwargs.pop("smoother_args")
                new_model = RNNModel(**self.model1.constructor_kwargs)
                self.model1 = fix_backward_compatibility_with_old_models(
                    self.model1, new_model
                )
                self.model1 = new_model.reconstruct(self.model1)
            else:
                w = self.model1
                rnn_cell_args = copy.deepcopy(
                    {
                        "ASettings": self.A1_args,
                        "KSettings": self.K1_args,
                        "CSettings": self.Cz1_args,
                    }
                )
                nft = self.ny + self.nu if self.has_Dyz and self.Dyz else self.nu
                self.model1 = RNNModel(
                    self.n1,
                    self.ny + self.nu,
                    self.block_samples,
                    self.batch_size,
                    ny_out=self.nz,
                    nft=nft,
                    LSTM_cell=hasattr(self, "LSTM_cell") and self.LSTM_cell,
                    out_dist=zDist,
                    missing_marker=self.missing_marker,
                    cell_args=rnn_cell_args,
                )
                w_rand = self.model1.model.get_weights()
                if len(w_rand) == len(w) + 1 and np.allclose(
                    w_rand[-1], np.zeros_like(w_rand[-1])
                ):  # Backward compatibility
                    w += [w_rand[-1]]
                self.model1.model.set_weights(w)
            self.model1.set_batch_size(1)

        if self.model1_Cy is not None and not isinstance(
            self.model1_Cy, (RegressionModel)
        ):
            if isinstance(self.model1_Cy, ReconstructionInfo) and hasattr(
                self.model1_Cy, "constructor_kwargs"
            ):
                model1_Cy_BU = self.model1_Cy
                w = model1_Cy_BU.weights
                self.model1_Cy = (
                    RegressionModel(**self.model1_Cy.constructor_kwargs)
                ).reconstruct(self.model1_Cy)
            else:
                w = self.model1_Cy
                if not hasattr(self, "model1_Cy_Full") or not self.model1_Cy_Full:
                    CyInputDim = self.n1 + self.nu
                else:
                    CyInputDim = self.n1 + self.n2 + self.nu
                reg_args = copy.deepcopy(self.Cy1_args)
                self.model1_Cy = RegressionModel(
                    CyInputDim, self.ny, missing_marker=self.missing_marker, **reg_args
                )
                self.model1_Cy.model.set_weights(w)

        if self.model1_Cy_fw is not None and not isinstance(
            self.model1_Cy_fw, (RegressionModel)
        ):
            self.model1_Cy_fw = (
                RegressionModel(**self.model1_Cy_fw.constructor_kwargs)
            ).reconstruct(self.model1_Cy_fw)

        if self.model1_FCy is not None and not isinstance(
            self.model1_FCy, (RegressionModel)
        ):
            self.model1_FCy = (
                RegressionModel(**self.model1_FCy.constructor_kwargs)
            ).reconstruct(self.model1_FCy)

        if self.model1_FCy_fw is not None and not isinstance(
            self.model1_FCy_fw, (RegressionModel)
        ):
            self.model1_FCy_fw = (
                RegressionModel(**self.model1_FCy_fw.constructor_kwargs)
            ).reconstruct(self.model1_FCy_fw)

        if self.n2 > 0 and not isinstance(self.model2, (RNNModel)):
            if isinstance(self.model2, ReconstructionInfo) and hasattr(
                self.model2, "constructor_kwargs"
            ):
                if (
                    "smooth_input" in self.model2.constructor_kwargs
                    and not self.model2.constructor_kwargs["smooth_input"]
                ):  # For backward compatibility
                    self.model2.constructor_kwargs.pop("smooth_input")
                    self.model2.constructor_kwargs.pop("smoother_args")
                new_model = RNNModel(**self.model2.constructor_kwargs)
                self.model2 = fix_backward_compatibility_with_old_models(
                    self.model2, new_model
                )
                self.model2 = new_model.reconstruct(self.model2)
            else:
                w = self.model2
                rnn_cell_args = copy.deepcopy(
                    {
                        "ASettings": self.A2_args,
                        "KSettings": self.K2_args,
                        "CSettings": self.Cy2_args,
                    }
                )
                self.model2 = RNNModel(
                    self.n2,
                    self.ny + self.n1 + self.nu,
                    self.block_samples,
                    self.batch_size,
                    ny_out=self.ny,
                    nft=self.nu,
                    LSTM_cell=hasattr(self, "LSTM_cell") and self.LSTM_cell,
                    has_prior_pred=True,  # From stage 1
                    out_dist=yDist,
                    missing_marker=self.missing_marker,
                    cell_args=rnn_cell_args,
                )
                w_rand = self.model2.model.get_weights()
                if len(w_rand) == len(w) + 1 and np.allclose(
                    w_rand[-1], np.zeros_like(w_rand[-1])
                ):  # Backward compatibility
                    w += [w_rand[-1]]
                self.model2.model.set_weights(w)
            self.model2.set_batch_size(1)

        if (
            self.nz > 0
            and self.model2_Cz is not None
            and not isinstance(self.model2_Cz, (RegressionModel))
        ):
            if isinstance(self.model2_Cz, ReconstructionInfo) and hasattr(
                self.model2_Cz, "constructor_kwargs"
            ):
                self.model2_Cz = (
                    RegressionModel(**self.model2_Cz.constructor_kwargs)
                ).reconstruct(self.model2_Cz)
            else:
                w = self.model2_Cz
                if not self.model2_Cz_Full:
                    CzInputDim = self.n2 + self.nu
                else:
                    CzInputDim = self.n1 + self.n2 + self.nu
                if self.has_Dyz and self.Dyz and (self.n1 == 0 or self.model2_Cz_Full):
                    CzInputDim += self.ny
                reg_args = copy.deepcopy(self.Cz2_args)
                reg_args["has_prior_pred"] = True  # From stage 1
                self.model2_Cz = RegressionModel(
                    CzInputDim, self.nz, missing_marker=self.missing_marker, **reg_args
                )
                self.model2_Cz.model.set_weights(w)

        if (
            self.nz > 0
            and self.model2_Cz_fw is not None
            and not isinstance(self.model2_Cz_fw, (RegressionModel))
        ):
            self.model2_Cz_fw = (
                RegressionModel(**self.model2_Cz_fw.constructor_kwargs)
            ).reconstruct(self.model2_Cz_fw)

    def saveToFile(self, filePath, saveTFModels=False):
        """Saves model in a pickle file

        Args:
            filePath (str): path to pickle file
            saveTFModels (bool): if True, will also save individual underlying TF models into full saved directories so that training for them can continue
        """
        if saveTFModels:
            self.saveTFModels(filePath)
        self.discardModels()  # Otherwise will be hard to save
        Path(filePath).parent.mkdir(parents=True, exist_ok=True)
        pickle_save(filePath, {"model": self})

    def loadFromFile(filePath):
        """Saves model from a pickle file

        Args:
            filePath (str): path to pickle file
        """
        fD = pickle_load(filePath)
        model = fD["model"]
        if hasattr(model, "restoreModels"):
            model.restoreModels()
        return model

    def getTFModelFilePaths(self, baseFilePath):
        """Returns the paths to save files for individual models based on the baseFilePath

        Args:
            baseFilePath (_type_): _description_

        Returns:
            _type_: _description_
        """
        out = {}
        model_names = [
            "model1",
            "model2",
            "model1_Cy",
            "model1_Cy_fw",
            "model1_FCy",
            "model1_FCy_fw",
            "model2_Cz",
            "model2_Cz_fw",
        ]
        for model_name in model_names:
            out[model_name] = f"{baseFilePath}_{model_name}.h5"
        return out

    def deleteTFModelFiles(self, baseFilePath):
        """Deletes the files for individul models based on the baseFilePath

        Args:
            baseFilePath (str): base path to save files.
        """
        for f in self.getTFModelFilePaths(baseFilePath).values():
            if os.path.exists(f):
                os.remove(f)

    def saveTFModels(self, baseFilePath):
        """Saves all tf model in complete form with optimizer state, etc in tf model directories

        Args:
            baseFilePath (str): base path to save files.
        """
        filePaths = self.getTFModelFilePaths(baseFilePath)
        model_names = [
            "model1",
            "model2",
            "model1_Cy",
            "model1_Cy_fw",
            "model1_FCy",
            "model1_FCy_fw",
            "model2_Cz",
            "model2_Cz_fw",
        ]
        for model_name in model_names:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                if model is not None:
                    model.save_to_file(filePaths[model_name])

    def loadTFModels(self, baseFilePath):
        """Loads all tf models in complete form with optimizer state, etc from tf model directory

        Args:
            baseFilePath (str): base path to save files.
        """
        filePaths = self.getTFModelFilePaths(baseFilePath)
        model_names = [
            "model1",
            "model2",
            "model1_Cy",
            "model1_Cy_fw",
            "model1_FCy",
            "model1_FCy_fw",
            "model2_Cz",
            "model2_Cz_fw",
        ]
        for model_name in model_names:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                if model is not None:
                    if os.path.exists(filePaths[model_name]):
                        model.load_from_file(filePaths[model_name])
                        setattr(self, model_name, model)
                    else:
                        logger.warning(f"Warning: {model_name} file not found")

    def getLSSM(self):
        """
        returns an LSSM with the same parameters as the fitted model
        """
        B_KDy1 = np.zeros((self.n1, self.nu))
        B_KDy2 = np.zeros((self.n2, self.nu))
        Dy = np.zeros((self.ny, self.nu))
        Dz = np.zeros((self.nz, self.nu))

        if hasattr(self, "has_Dyz") and self.has_Dyz:
            Dyz = np.zeros((self.nz, self.ny))

        # RNN1
        if self.n1 > 0:
            # allWeights1 = self.model1.model.get_weights()
            allWeights1 = [
                self.model1.rnn.cell.A.get_weights()[0],
                self.model1.rnn.cell.K.get_weights()[0],
                self.model1.rnn.cell.C.get_weights()[0],
                self.model1.rnn.cell.initial_state.numpy(),
            ]
            if (
                "unifiedAK" in self.model1.cell_args
                and self.model1.cell_args["unifiedAK"]
            ):
                allWeights1BU = copy.deepcopy(allWeights1)
                allWeights1 = [
                    allWeights1BU[0][: self.n1, :],  # The A_KCy part
                    allWeights1BU[0][self.n1 :, :],  # The K part
                    allWeights1BU[1],
                ]
            A_KCy = allWeights1[0].T  # n1 x n1
            K = allWeights1[1].T  # n1 x (ny+nu)
            CzDz = allWeights1[2].T  # nz x (n1+nu)
            x0 = allWeights1[3]  # n1
            if self.nu > 0:
                B_KDy1 = K[:, self.ny :]  # n1 x nu
                K = K[:, : self.ny]  # n1 x ny

            if (
                self.nz > 0
                and self.model2_Cz is not None
                and self.n2 == 0
                and self.model2_Cz_Full
            ):  # Use the regression for CzDz instead of the one from RNN
                CzDz = self.model2_Cz.model.get_weights()[0].T  # nz x (n1+n2+nu)
                # CzDz_bias = self.model2_Cz.model.get_weights()[1]

            if hasattr(self, "has_Dyz") and self.has_Dyz:
                Dyz = CzDz[:, self.n1 : (self.n1 + self.ny)]
                CzDz = np.concatenate(
                    (CzDz[:, : self.n1], CzDz[:, (self.n1 + self.ny) :]), axis=1
                )

        if self.model1_FCy is not None and self.model1_Cy_Full:
            CyDy = self.model1_FCy.model.get_weights()[0].T  # ny x (n1+n2+nu)
            # CyDy_bias = self.model1_Cy.model.get_weights()[1] # The bias term
        elif self.model1_Cy is not None:
            CyDy = self.model1_Cy.model.get_weights()[0].T  # ny x (n1+nu)
            # CyDy_bias = self.model1_Cy.model.get_weights()[1] # The bias term

        # RNN2
        if self.n2 > 0:
            # allWeights2 = self.model2.model.get_weights()
            allWeights2 = [
                self.model2.rnn.cell.A.get_weights()[0],
                self.model2.rnn.cell.K.get_weights()[0],
                self.model2.rnn.cell.C.get_weights()[0],
                self.model2.rnn.cell.initial_state.numpy(),
            ]
            if (
                "unifiedAK" in self.model2.cell_args
                and self.model2.cell_args["unifiedAK"]
            ):
                allWeights2BU = copy.deepcopy(allWeights2)
                allWeights2 = [
                    allWeights2BU[0][: self.n2, :],  # The A_KCy2 part
                    allWeights2BU[0][self.n2 :, :],  # The K2 part
                    allWeights2BU[1],
                ]
            A_KCy2 = allWeights2[0].T  # n2 x n2
            K2 = allWeights2[1].T  # n2 x (n1+ny+nu)
            CyDy2 = allWeights2[2].T  # ny x (n2+nu)
            x0_2 = allWeights2[3]  # n2
            if not self.model1_Cy_Full:
                if self.model1_Cy is None:
                    CyDy = CyDy2  # ny x (n2+nu)
                else:
                    Cy1 = CyDy[:, : self.n1]
                    Cy2 = CyDy2[:, : self.n2]
                    Dy = (
                        CyDy[:, self.n1 :] + CyDy2[:, self.n2 :]
                    )  # Add up any feedthrough from two stages
                    CyDy = np.concatenate((Cy1, Cy2, Dy), axis=1)  # ny x (n1+n2+nu)

            if self.nu > 0:
                B_KDy2 = K2[:, (self.ny + self.n1) :]  # n2 x nu
                K2 = K2[:, : (self.ny + self.n1)]  # n2 x (n1+ny)

            if self.nz > 0 and self.model2_Cz is not None:
                CzDz2 = self.model2_Cz.model.get_weights()[
                    0
                ].T  # nz x n2 or nz x (n1+n2+nu)
                # CzDz2_bias = self.model2_Cz.model.get_weights()[1]

                if (
                    hasattr(self, "has_Dyz")
                    and self.has_Dyz
                    and (self.n1 == 0 or self.model2_Cz_Full)
                ):
                    Cz2Size = self.n2 if not self.model2_Cz_Full else self.n1 + self.n2
                    Dyz = CzDz2[:, Cz2Size : (Cz2Size + self.ny)]
                    CzDz2 = np.concatenate(
                        (CzDz2[:, :Cz2Size], CzDz2[:, (Cz2Size + self.ny) :]), axis=1
                    )
            else:
                if not self.model2_Cz_Full:
                    CzDz2 = np.zeros((self.nz, self.n2 + self.nu))
                else:
                    CzDz2 = np.zeros((self.nz, self.n1 + self.n2 + self.nu))

            if (
                self.model2_Cz is None
                and (not hasattr(self, "allow_nonzero_Cz2") or self.allow_nonzero_Cz2)
            ) or (self.model2_Cz_Full or self.n1 == 0):
                CzDz = CzDz2
            else:
                Cz1 = CzDz[:, : self.n1]
                Cz2 = CzDz2[:, : self.n2]
                Dz = (
                    CzDz[:, self.n1 :] + CzDz2[:, self.n2 :]
                )  # Add up any feedthrough from two stages
                CzDz = np.concatenate((Cz1, Cz2, Dz), axis=1)  # nz x (n1+n2+nu)

            if self.n1 == 0:
                A_KCy = A_KCy2  # n2 x n2
                K = K2  # n2 x ny
                x0 = x0_2
            else:
                K2BU = K2  # n2 x (n1+ny)
                A_KCy21 = K2BU[:, : self.n1]  # n2 x n1
                A_KCy12 = np.zeros_like(A_KCy21).T
                K2 = K2BU[:, self.n1 :]  # n2 x ny

                Cy = CyDy[:, : self.n1]
                Cy2 = CyDy[:, self.n1 : self.nx]

                A_KCy = np.concatenate(
                    (
                        np.concatenate((A_KCy, A_KCy12), axis=1),
                        np.concatenate((A_KCy21, A_KCy2), axis=1),
                    ),
                    axis=0,
                )

                K = np.concatenate((K, K2), axis=0)  # nx x ny
                x0 = np.concatenate((x0, x0_2))  # nx

        if self.model1_Cy_fw is not None:
            logger.warning(
                "model1_Cy_fw exists in the model but will be ignored in getLSSM, since it is meant to be a subset of model1_Cy"
            )
        if self.model2_Cz_fw is not None:
            logger.warning(
                "model2_Cz_fw exists in the model but will be ignored in getLSSM, since it is meant to be a subset of model2_Cz"
            )

        if hasattr(self, "YPrepMap") and self.YPrepMap is not None:
            YPrepMapW = self.YPrepMap.get_overall_W()
            if YPrepMapW is not None:
                K = K @ YPrepMapW
                CyDy = np.linalg.pinv(YPrepMapW) @ CyDy

        Cy = CyDy[:, : self.nx]
        Dy = CyDy[:, self.nx :]
        Cz = CzDz[:, : self.nx]
        Dz = CzDz[:, self.nx :]
        A = A_KCy + np.matmul(K, Cy)
        B_KDy = np.concatenate((B_KDy1, B_KDy2), axis=0)
        B = B_KDy + np.matmul(K, Dy)

        if hasattr(self, "UPrepMap") and self.UPrepMap is not None:
            UPrepMapW = self.UPrepMap.get_overall_W()
            if UPrepMapW is not None:
                B = B @ UPrepMapW

        params = {
            "A": A,
            "B": B,
            "C": Cy,
            "D": Dy,
            "K": K,
            "Cz": Cz,
            "Dz": Dz,
            "innovCov": self.YErrCov,
            "x0": x0,
        }
        if hasattr(self, "has_Dyz") and self.has_Dyz:
            params["Dyz"] = Dyz
        s = LSSM(params=params)
        s.zDims = 1 + np.arange(self.n1)
        setattr(s, "info", {})
        s.info["logs"] = self.logs if hasattr(self, "logs") else {}
        return s

    def setToLSSM(
        self,
        sys,
        model1_Cy_Full=True,
        model2_Cz_Full=True,
        allow_nonzero_Cz2=True,
        A_KC_is_block_diagonal=False,
        props={},
        YType=None,
        ZType=None,
        A_args={},
        K_args={},
        Cy_args={},
        Cz_args={},  # Both stage 1 and 2 params
        A1_args=None,
        K1_args=None,
        Cy1_args=None,
        Cz1_args=None,  # Stage 1 params
        A2_args=None,
        K2_args=None,
        Cy2_args=None,
        Cz2_args=None,  # Stage 2 params
        x0=None,  # Initial state
        stateful=True,  # Whether to use stateful LSSM
        enable_forward_pred=True,  # If true, will enable forward prediction by having a separate set of Afw,Kfw,Cfw parameters
        ignore_Zero_A_KC_topRight=False,  # If True, will reduce the error to a warning
        ignore_Zero_A_topRight=False,  # If True, will reduce the error to a warning
        bidirectional=False,  # Should the RNN be bidirectional or not
        steps_ahead=None,  # List of multi-step ahead predictions to equip the network with
    ):
        """Sets the DPADModel to be equal to an LSSM.

        Args:
            sys ([type]): the input LSSM.
            model1_Cy_Full (bool, optional): If true, will model stage 1 and 2 Cy's together
                    in the model1_Cy. Defaults to True.
            model2_Cz_Full (bool, optional): If true, will model stage 1 and 2 Cz's together
                    in the model2_Cz. Defaults to True.
            allow_nonzero_Cz2 (bool, optional): If True, will allow stage 2 to have a Cz.
                    Defaults to True. Will always be true if model2_Cz_Full is true.
            A_KC_is_block_diagonal (bool, optional): Input LSSM must be in predictor form and
                have block diagonal A_KC. If this argument is False, the input LSSM will first
                be converted to have block diagonal A_KC. Set to True if LSSM already has a
                block diagonal A_KC. Defaults to False.
            props (dict, optional): Additional properties to set for the model. Defaults to {}.

        Returns:
            E (matrix): Similarity tranform matrix used to make LSSM have block diagonal A_KC.
        """
        if A1_args is None:
            A1_args = A_args
        if A2_args is None:
            A2_args = A_args
        if K1_args is None:
            K1_args = K_args
        if K2_args is None:
            K2_args = K_args
        if Cy1_args is None:
            Cy1_args = Cy_args
        if Cy2_args is None:
            Cy2_args = Cy_args
        if Cz1_args is None:
            Cz1_args = Cz_args
        if Cz2_args is None:
            Cz2_args = Cz_args

        _, _, yDist = self.prep_observation_for_training(None, YType)
        _, _, zDist = self.prep_observation_for_training(None, ZType)
        (
            self.A1_args,
            self.K1_args,
            self.Cy1_args,
            self.Cz1_args,
            self.A2_args,
            self.K2_args,
            self.Cy2_args,
            self.Cz2_args,
        ) = self.add_default_param_args(
            A1_args,
            K1_args,
            Cy1_args,
            Cz1_args,
            A2_args,
            K2_args,
            Cy2_args,
            Cz2_args,
            yDist,
            zDist,
        )

        if (
            steps_ahead is not None
            and (len(steps_ahead) > 1 or steps_ahead[0] != 1)
            and not enable_forward_pred
        ):
            raise (
                Exception(
                    "enable_forward_pred must be True if steps_ahead includes numbers other than 1"
                )
            )

        s = copy.deepcopy(sys)
        if not hasattr(self, "nx"):
            self.nx = s.state_dim
        if not hasattr(self, "ny"):
            self.ny = s.output_dim
        if not hasattr(self, "nu"):
            self.nu = s.input_dim

        if not hasattr(self, "n1"):
            if hasattr(s, "zDims") and isinstance(s.zDims, np.ndarray):
                self.n1 = s.zDims.size
            else:
                self.n1 = self.nx
        if self.n1 > self.nx:
            self.n1 = self.nx
        self.n2 = self.nx - self.n1

        if model2_Cz_Full:
            allow_nonzero_Cz2 = True

        if hasattr(s, "Dyz"):
            self.has_Dyz = True
            Dyz = s.Dyz
        else:
            self.has_Dyz = False

        steps_ahead, _, steps_ahead_model1, _, _ = self.get_model_steps_ahead(
            steps_ahead
        )

        self.steps_ahead = steps_ahead
        need_fw_reg_models = np.any(np.array(steps_ahead) != 1) and (
            self.nu > 0 or self.has_Dyz
        )

        self.bidirectional = bidirectional

        A_KC_topRight = s.A_KC[: self.n1, self.n1 :]
        topRightIsZero = np.linalg.norm(A_KC_topRight) / np.linalg.norm(s.A_KC) < 1e-8
        if (
            self.n1 > 0
            and self.n2 > 0
            and not A_KC_is_block_diagonal
            and not topRightIsZero
        ):
            E = s.makeA_KCBlockDiagonal()  # Convert to block diagonal A-KC
        else:
            E = np.eye(self.nx)

        A = s.A
        A_KC = s.A_KC
        K = s.K
        if hasattr(s, "Cy"):
            Cy = s.Cy
        elif hasattr(s, "C"):
            Cy = s.C
        if hasattr(s, "Cz"):
            Cz = s.Cz
            self.nz = Cz.shape[0]
        else:
            Cz = np.zeros((self.nz, self.nx))
        if hasattr(s, "B_KD"):
            B_KD = s.B_KD
        else:
            B_KD = np.empty((self.nx, 0))

        if hasattr(s, "D"):
            Dy = s.D
        else:
            Dy = np.zeros((self.ny, self.nu))
        if hasattr(s, "Dz"):
            Dz = s.Dz
        else:
            Dz = np.zeros((self.nz, self.nu))

        if hasattr(s, "x0") and x0 is None:
            x0 = s.x0

        if self.n1 > 0 and self.n2 > 0:
            A_KC_topRight = A_KC[: self.n1, self.n1 :]
            if np.linalg.norm(A_KC_topRight) / np.linalg.norm(A_KC) > 1e-6:
                err = "Top right {}x{} block of A_KC must be almost zero, otherwise DPAD model with n1={} will NOT be equivalent to the original LSSM.".format(
                    self.n1, self.n2, self.n1
                )
                if not ignore_Zero_A_KC_topRight:
                    raise (Exception(err))
                else:
                    warnings.warn(err)
            if enable_forward_pred:
                A_topRight = A[: self.n1, self.n1 :]
                if np.linalg.norm(A_topRight) / np.linalg.norm(A) > 1e-6:
                    err = "Top right {}x{} block of A must be almost zero to enable forward prediction, otherwise DPAD model with n1={} will NOT be equivalent to the original LSSM in terms of forward prediction.".format(
                        self.n1, self.n2, self.n1
                    )
                    if not ignore_Zero_A_topRight:
                        raise (Exception(err))
                    else:
                        warnings.warn(err)

        # Cz model
        self.model2_Cz_Full = model2_Cz_Full
        if (
            self.model2_Cz_Full or self.n1 == 0
        ):  # Dz needs to be taken into account in model2_Cz
            CzThis = Cz
            DzThis = Dz
            DzRNN1 = np.zeros(Dz.shape)
        else:  # Dz will be incorporated in RNN1
            CzThis = Cz[:, self.n1 :]
            if CzThis.size > 0:
                DzThis = np.zeros(Dz.shape)
            else:  # No need for model2_Cz
                DzThis = np.zeros((Dz.shape[0], 0))
            DzRNN1 = Dz
        if self.has_Dyz and (
            self.n1 == 0 or self.model2_Cz_Full
        ):  # Dyz will be incorporated in model2_Cz
            DzThis = np.concatenate((Dyz, DzThis), axis=1)
        CzDz = np.concatenate((CzThis, DzThis), axis=1)
        if not allow_nonzero_Cz2 and not model2_Cz_Full and self.n1 > 0 and self.n2 > 0:
            if (
                np.linalg.norm(CzDz) / np.linalg.norm(np.concatenate((Cz, Dz), axis=1))
                < 1e-6
            ):
                CzDz = np.zeros((CzDz.shape[0], 0))
            elif CzDz.size > 0:
                raise (
                    Exception(
                        "CzDz must be all zeros, otherwise DPAD model with n1={} will NOT be equivalent to the original LSSM.".format(
                            self.n1
                        )
                    )
                )
        reg_args = copy.deepcopy(self.Cz2_args)
        reg_args["has_prior_pred"] = True  # From stage 1
        if CzDz.size > 0:
            self.model2_Cz = RegressionModel(
                CzDz.shape[-1], self.nz, missing_marker=self.missing_marker, **reg_args
            )
            self.model2_Cz.model.set_weights([CzDz.T])
        else:
            self.model2_Cz = None

        if need_fw_reg_models and CzThis.size > 0:
            self.model2_Cz_fw = RegressionModel(
                CzThis.shape[-1],
                self.nz,
                missing_marker=self.missing_marker,
                **reg_args,
            )
            self.model2_Cz_fw.model.set_weights([CzThis.T])
        else:
            self.model2_Cz_fw = None

        # RNN1
        self.stateful = stateful
        if self.n1 > 0:
            K1B1 = np.concatenate((K[: self.n1, :], B_KD[: self.n1, :]), axis=1)
            if self.has_Dyz:  # Dyz will be incorporated in RNN1
                DzRNN1 = np.concatenate((Dyz, DzRNN1), axis=1)
                nft = self.ny + self.nu
            else:
                nft = self.nu
            Cz1Dz = np.concatenate((Cz[:, : self.n1], DzRNN1), axis=1)

            w = {"A": A_KC[: self.n1, : self.n1].T, "K": K1B1.T, "C": Cz1Dz.T}

            rnn_cell_args = copy.deepcopy(
                {
                    "ASettings": self.A1_args,
                    "KSettings": self.K1_args,
                    "CSettings": self.Cz1_args,
                }
            )
            if x0 is not None:
                rnn_cell_args["learn_initial_state"] = True
                x0_1 = x0[: self.n1]
                w["initial_state"] = x0_1

            if steps_ahead is not None and (
                len(steps_ahead) > 1 or steps_ahead[0] != 1
            ):
                w["Afw"] = A[: self.n1, : self.n1].T
                if nft > 0:
                    w["Cfw"] = Cz[:, : self.n1].T
            self.model1 = RNNModel(
                self.n1,
                self.ny + self.nu,
                self.block_samples,
                self.batch_size,
                ny_out=self.nz,
                nft=nft,
                enable_forward_pred=enable_forward_pred,
                steps_ahead=steps_ahead_model1,
                missing_marker=self.missing_marker,
                cell_args=rnn_cell_args,
                stateful=self.stateful,
            )
            # self.model1.model.set_weights(w)
            self.model1.set_cell_weights(w)
            self.model1.set_batch_size(1)
        else:
            self.model1 = None

        # Cy model
        self.model1_Cy_Full = model1_Cy_Full
        Cy1This = Cy[:, : self.n1]
        if (
            self.model1_Cy_Full or self.n2 == 0
        ):  # Dy will be taken into account in model1_Cy
            CyThis = Cy
            DyThis = Dy
            DyRNN2 = np.zeros(Dy.shape)
        else:  # Dy will be incorporated in RNN2
            CyThis = Cy1This
            if CyThis.size > 0:
                DyThis = np.zeros(Dy.shape)
            else:  # No need for model1_Cy
                DyThis = np.zeros((Dy.shape[0], 0))
            DyRNN2 = Dy
        CyDy = np.concatenate((CyThis, DyThis), axis=1)
        Cy1Dy = np.concatenate((Cy1This, DyThis), axis=1)
        reg_args = copy.deepcopy(self.Cy1_args)
        if CyDy.size > 0:
            self.model1_Cy = RegressionModel(
                Cy1Dy.shape[-1], self.ny, missing_marker=self.missing_marker, **reg_args
            )
            self.model1_Cy.model.set_weights([Cy1Dy.T])
            if model1_Cy_Full:
                self.model1_FCy = RegressionModel(
                    CyDy.shape[-1],
                    self.ny,
                    missing_marker=self.missing_marker,
                    **reg_args,
                )
                self.model1_FCy.model.set_weights([CyDy.T])
            else:
                self.model1_FCy = None
        else:
            self.model1_Cy, self.model1_FCy = None, None

        if need_fw_reg_models and CyThis.size > 0:
            self.model1_Cy_fw = RegressionModel(
                Cy1This.shape[-1],
                self.ny,
                missing_marker=self.missing_marker,
                **reg_args,
            )
            self.model1_Cy_fw.model.set_weights([Cy1This.T])
            if model1_Cy_Full:
                self.model1_FCy_fw = RegressionModel(
                    CyThis.shape[-1],
                    self.ny,
                    missing_marker=self.missing_marker,
                    **reg_args,
                )
                self.model1_FCy_fw.model.set_weights([CyThis.T])
            else:
                self.model1_FCy_fw = None
        else:
            self.model1_Cy_fw, self.model1_FCy_fw = None, None

        # RNN2
        if self.n2 > 0:
            A_KC21 = A_KC[self.n1 :, : self.n1]
            K2B2 = np.concatenate((K[self.n1 :, :], B_KD[self.n1 :, :]), axis=1)
            Cy2Dy = np.concatenate((Cy[:, self.n1 :], DyRNN2), axis=1)

            w = {
                "A": A_KC[self.n1 :, self.n1 :].T,
                "K": np.concatenate((A_KC21, K2B2), axis=1).T,
                "C": Cy2Dy.T,
            }

            rnn_cell_args = copy.deepcopy(
                {
                    "ASettings": self.A2_args,
                    "KSettings": self.K2_args,
                    "CSettings": self.Cy2_args,
                }
            )
            if x0 is not None:
                rnn_cell_args["learn_initial_state"] = True
                x0_2 = x0[self.n1 :]
                w["initial_state"] = x0_2

            if steps_ahead is not None and (
                len(steps_ahead) > 1 or steps_ahead[0] != 1
            ):
                w["Afw"] = A[self.n1 :, self.n1 :].T
                if self.nu > 0:
                    w["Cfw"] = Cy[:, self.n1 :].T
                if self.n1 > 0:
                    w["Kfw"] = (A[self.n1 :, : self.n1]).T

            self.model2 = RNNModel(
                self.n2,
                self.ny + self.n1 + self.nu,
                self.block_samples,
                self.batch_size,
                ny_out=self.ny,
                nft=self.nu,
                n1_in=self.n1,
                has_prior_pred=True,  # From stage 1
                enable_forward_pred=enable_forward_pred,
                steps_ahead=steps_ahead,
                missing_marker=self.missing_marker,
                cell_args=rnn_cell_args,
                stateful=self.stateful,
            )
            # self.model2.model.set_weights(w)
            self.model2.set_cell_weights(w)
            self.model2.set_batch_size(1)
        else:
            self.model2 = None

        self.logs = {}

        self.YErrCov = s.innovCov

        self.YType = YType
        self.ZType = ZType

        for k, v in props.items():
            setattr(self, k, v)

        return E

    def predict(self, Y, U=None, x0=None):
        """Runs prediction for a given input data

        Args:
            Y (np.array): main input time series. Dimensions are sample x ny.
            U (np.array, optional): secondary input time series U. Defaults to None. Dimensions are sample x nu.
            x0 (np.array, optional): if not None, will replace 0 as the initial state. Defaults to None.

        Returns:
            allZp (np.array): prediction of the output signal z
            allYp (np.array), self-prediction of the input signal y
            allXp (np.array): estimated latent states
        """
        eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        if hasattr(self, "yPrepModel") and self.yPrepModel is not None:
            Y = np.array(self.yPrepModel.predict(Y))

        if hasattr(self, "YPrepMap") and self.YPrepMap is not None:
            Y = self.YPrepMap.apply(Y.T).T
        if hasattr(self, "UPrepMap") and self.UPrepMap is not None and U is not None:
            U = self.UPrepMap.apply(U.T).T

        # TEMP REFERENCE
        try:
            sId = self.getLSSM()
            allZpLSSM, allYpLSSM, allXpLSSM = sId.predict(Y, U=U, x0=x0)
        except Exception as e:
            # logger.info(e)
            pass

        Ndat = Y.shape[0]

        if U is not None:
            raise (Exception(INPUT_MSG))

        if U is None and self.nu > 0:
            U = np.zeros((Ndat, self.nu))
        if U is not None:
            UT = U.T
        else:
            UT = None

        nx = self.nx if not self.bidirectional else 2 * self.nx
        n1 = self.n1 if not self.bidirectional else 2 * self.n1

        allXp = np.zeros((Ndat, nx))
        allYp = None
        allZp = None

        if hasattr(self, "steps_ahead") and self.steps_ahead is not None:
            steps_ahead = self.steps_ahead
        else:
            steps_ahead = [1] if not self.bidirectional else [0]
        (
            steps_ahead,
            _,
            steps_ahead_model1,
            _,
            model1_orig_step_inds,
        ) = self.get_model_steps_ahead(steps_ahead)
        need_fw_reg_models = (
            np.any(np.array(steps_ahead) != 1)
            and (self.nu > 0 or self.has_Dyz)
            and (
                not hasattr(self, "use_feedthrough_in_fw")
                or not self.use_feedthrough_in_fw
            )
        )

        multi_step_with_data_gen = (
            hasattr(self, "multi_step_with_data_gen") and self.multi_step_with_data_gen
        )
        if multi_step_with_data_gen:
            steps_ahead_model1_backup = copy.copy(steps_ahead_model1)
            steps_ahead_backup = copy.copy(steps_ahead)
            model1_orig_step_inds_backup = copy.copy(model1_orig_step_inds)
            (
                steps_ahead,
                _,
                steps_ahead_model1,
                _,
                model1_orig_step_inds,
            ) = self.get_model_steps_ahead([1])

        allXp_steps = [np.zeros((Ndat, nx)) for s in steps_ahead]
        allYp_steps = [None for s in steps_ahead]
        allZp_steps = [None for s in steps_ahead]
        allXp_internal_steps = [None for s in steps_ahead]

        if U is not None:
            YU = np.concatenate((Y, U), axis=1)
        else:
            YU = Y
        if n1 > 0:
            if self.has_Dyz:
                FT_in = YU.T
            else:
                FT_in = UT
            if x0 is not None:
                initial_state = x0[:n1].T
            else:
                initial_state = None
            preds1 = self.model1.predict(
                YU.T,
                FT_in=FT_in,
                initial_state=initial_state,
                return_internal_states=multi_step_with_data_gen,
            )
            allXp1_steps = preds1[
                : len(steps_ahead_model1)
            ]  # if steps_ahead is not None else preds1[0]
            allZp1_steps = preds1[
                len(steps_ahead_model1) : 2 * len(steps_ahead_model1)
            ]  # if steps_ahead is not None else preds1[1]
            if multi_step_with_data_gen:
                allXp1_internal_steps = preds1[
                    3 * len(steps_ahead_model1) : 4 * len(steps_ahead_model1)
                ]  # Useful for LSTM

            allXp1step, allZp1step = (
                allXp1_steps[0],
                allZp1_steps[0],
            )  # This will be used for the next stage

            allZp_steps = [None for s in steps_ahead]
            for saInd in range(len(steps_ahead)):
                allXp_steps[saInd][:, :n1] = allXp1_steps[
                    model1_orig_step_inds[saInd]
                ].T
                allZp_steps[saInd] = allZp1_steps[model1_orig_step_inds[saInd]]
                if multi_step_with_data_gen:
                    allXp_internal_steps[saInd] = allXp1_internal_steps[
                        model1_orig_step_inds[saInd]
                    ].T

            if self.n2 > 0:
                Y_in_res = np.concatenate((allXp1step, Y.T), axis=0)
            if not self.model1_Cy_Full:
                if self.model1_Cy is not None:
                    for saInd in range(len(steps_ahead)):
                        allXp1_this_step = allXp1_steps[model1_orig_step_inds[saInd]]
                        sa = steps_ahead[saInd]
                        if need_fw_reg_models and sa > 1:
                            allYp1 = self.model1_Cy_fw.predict(allXp1_this_step)
                        else:
                            if U is not None:
                                UShift = np.concatenate(
                                    (U, np.nan * np.ones((sa - 1, U.shape[-1]))), axis=0
                                )[(sa - 1) :, :]
                                allXp1U = np.concatenate((allXp1_this_step, UShift.T))
                            else:
                                allXp1U = allXp1_this_step
                            allYp1 = self.model1_Cy.predict(allXp1U)
                        allYp_steps[saInd] = allYp1
                else:
                    if self.n2 > 0:
                        raise (Exception("Model does not have model1_Cy!"))

            """
            # Plot training & validation loss values
            plt.plot(allZpLSSM[:, :1], label='KF')
            plt.plot(allZp1[:1, :].T, label='RNN', linestyle='dashed')
            plt.xlabel('Time')
            plt.legend()
            plt.show()
            """
        else:
            Y_in_res = Y.T

        if self.n2 > 0:
            if U is not None:
                Y_in_resU = np.concatenate((Y_in_res, U.T), axis=0)
            else:
                Y_in_resU = Y_in_res
            if x0 is not None:
                initial_state = x0[n1:].T
            else:
                initial_state = None
            n1_in = allXp1_steps if n1 > 0 else None
            preds2 = self.model2.predict(
                Y_in_resU,
                FT_in=UT,
                n1_in=n1_in,
                prior_pred=allYp_steps,
                initial_state=initial_state,
                prior_pred_shift_by_one=True,
                return_internal_states=multi_step_with_data_gen,
            )

            allXp2_steps = preds2[: len(steps_ahead)]
            allYpF_steps = preds2[len(steps_ahead) : 2 * len(steps_ahead)]
            if multi_step_with_data_gen:
                allXp2_internal_steps = preds2[
                    3 * len(steps_ahead) : 4 * len(steps_ahead)
                ]  # Useful for LSTM

            allXp2, allYpF = allXp2_steps[0], allYpF_steps[0]

            for saInd in range(len(steps_ahead)):
                allXp_steps[saInd][:, n1:] = allXp2_steps[saInd].T
                if multi_step_with_data_gen:
                    if allXp_internal_steps[saInd] is None:
                        allXp_internal_steps[saInd] = allXp2_internal_steps[saInd].T
                    else:
                        allXp_internal_steps[saInd] = np.concatenate(
                            (
                                allXp_internal_steps[saInd],
                                allXp2_internal_steps[saInd].T,
                            ),
                            axis=1,
                        )

            allYp_steps = list(allYpF_steps)

        if self.nz > 0 and self.model2_Cz is not None:
            UOrYU = (
                YU if self.has_Dyz and (n1 == 0 or self.model2_Cz_Full) else U
            )  # Will be the required feedthrough input for model2_Cz
            for saInd, allXp_this_step in enumerate(allXp_steps):
                sa = steps_ahead[saInd]
                if UOrYU is not None:
                    UOrYUShift = np.concatenate(
                        (UOrYU, np.nan * np.ones((sa - 1, UOrYU.shape[-1]))), axis=0
                    )[(sa - 1) :, :]
                if not self.model2_Cz_Full:
                    allXp2_this_step = allXp2_steps[saInd]
                    allZp_this_step = allZp_steps[saInd]
                    if need_fw_reg_models and sa > 1:
                        allZpF = self.model2_Cz_fw.predict(
                            allXp2_this_step, prior_pred=allZp_this_step
                        )
                    else:
                        if UOrYU is not None:
                            allXp2U = np.concatenate(
                                (allXp2_this_step.T, UOrYUShift), axis=1
                            )
                        else:
                            allXp2U = allXp2_this_step.T
                        allZpF = self.model2_Cz.predict(
                            allXp2U.T, prior_pred=allZp_this_step
                        )
                else:
                    if need_fw_reg_models and sa > 1:
                        allZpF = self.model2_Cz_fw.predict(
                            allXp_this_step.T, prior_pred=None
                        )
                    else:
                        if UOrYU is not None:
                            allXpU = np.concatenate(
                                (allXp_this_step, UOrYUShift), axis=1
                            )
                        else:
                            allXpU = allXp_this_step
                        allZpF = self.model2_Cz.predict(allXpU.T, prior_pred=None)
                allZp_steps[saInd] = allZpF
            allZp = allZp_steps[0]

        if self.model1_FCy is not None and self.model1_Cy_Full:
            for saInd, allXp_this_step in enumerate(allXp_steps):
                sa = steps_ahead[saInd]
                if need_fw_reg_models and sa > 1:
                    allYpF = self.model1_FCy_fw.predict(
                        allXp_this_step.T, prior_pred=None
                    )
                else:
                    if U is not None:
                        UShift = np.concatenate(
                            (U, np.nan * np.ones((sa - 1, U.shape[-1]))), axis=0
                        )[(sa - 1) :, :]
                        allXpU = np.concatenate((allXp_this_step, U), axis=1)
                    else:
                        allXpU = allXp_this_step
                    allYpF = self.model1_FCy.predict(allXpU.T, prior_pred=None)
                allYp_steps[saInd] = allYpF

        if multi_step_with_data_gen:  # The second approach for forecasting
            # Propagate without adding noise
            noise_samples = (
                self.multi_step_data_gen_noise_samples
                if hasattr(self, "multi_step_data_gen_noise_samples")
                else 0
            )
            if noise_samples > 0:
                allXp_steps_ns, allYp_steps_ns, allZp_steps_ns = [], [], []
                for nsi in range(noise_samples):
                    (
                        allXp_steps_this,
                        allYp_steps_this,
                        allZp_steps_this,
                    ) = self.propagate_prediction_forward_with_data_gen(
                        allXp_steps,
                        allXp_internal_steps,
                        allYp_steps,
                        allZp_steps,
                        steps_ahead_model1_backup,
                        steps_ahead_backup,
                        noise_scale=0.1 if nsi > 0 else 0,
                    )
                    allXp_steps_ns.append(allXp_steps_this)
                    allYp_steps_ns.append(allYp_steps_this)
                    allZp_steps_ns.append(allZp_steps_this)
                logger.info(
                    f"Averaging DPAD forecasting predictions across {noise_samples} noise realizations"
                )
                allXp_steps = list(np.mean(np.array(allXp_steps_ns), axis=0))
                allYp_steps = list(np.mean(np.array(allYp_steps_ns), axis=0))
                allZp_steps = list(np.mean(np.array(allZp_steps_ns), axis=0))
            else:
                (
                    allXp_steps,
                    allYp_steps,
                    allZp_steps,
                ) = self.propagate_prediction_forward_with_data_gen(
                    allXp_steps,
                    allXp_internal_steps,
                    allYp_steps,
                    allZp_steps,
                    steps_ahead_model1_backup,
                    steps_ahead_backup,
                )

        if self.ny > 0:
            for saInd, allYp in enumerate(allYp_steps):
                if allYp is not None:
                    if (
                        hasattr(self, "yPrepModel")
                        and self.yPrepModel is not None
                        and hasattr(self.yPrepModel, "inverse_transform")
                    ):
                        allYp = np.array(self.yPrepModel.inverse_transform(allYp.T)).T

                    if hasattr(self, "YPrepMap") and self.YPrepMap is not None:
                        allYp = self.YPrepMap.apply_inverse(allYp)

                    if len(allYp.shape) == 2:
                        allYp = allYp.T
                    else:
                        allYp = allYp.transpose([1, 0, 2])
                    allYp_steps[saInd] = allYp
        if self.nz > 0:
            for saInd, allZp in enumerate(allZp_steps):
                if hasattr(self, "ZPrepMap") and self.ZPrepMap is not None:
                    allZp = self.ZPrepMap.apply_inverse(allZp)

                if len(allZp.shape) == 2:
                    allZp = allZp.T
                else:
                    allZp = allZp.transpose([1, 0, 2])
                allZp_steps[saInd] = allZp

        set_global_tf_eagerly_flag(eagerly_flag_backup)
        return tuple(allZp_steps) + tuple(allYp_steps) + tuple(allXp_steps)

    def propagate_prediction_forward_with_data_gen(
        self,
        allXp_steps,
        allXp_internal_steps,
        allYp_steps,
        allZp_steps,
        steps_ahead_model1,
        steps_ahead,
        noise_scale=0,
    ):
        # Pass current predictions as new observations and do one more recursion from both stages
        allXp_all_steps, allYp_all_steps, allZp_all_steps = [], [], []
        for step_ahead in steps_ahead_model1:
            if step_ahead == 1:
                next_states = np.array(allXp_steps[0])
                next_states_internal = np.array(allXp_internal_steps[0])
                next_Yp = np.array(allYp_steps[0].T)
                next_Zp = np.array(allZp_steps[0].T)
            else:
                next_states = np.zeros_like(latest_states)
                next_states_internal = np.zeros_like(latest_states_internal)
                n1_internal = (
                    2 * self.n1 if self.n1 > 0 and self.model1.LSTM_cell else self.n1
                )
                n2_internal = (
                    2 * self.n2 if self.n2 > 0 and self.model2.LSTM_cell else self.n2
                )
                next_Yp = np.zeros_like(latest_Yp)
                next_Zp = np.zeros_like(latest_Zp)
                if self.n1 > 0:
                    (
                        next_states1,
                        next_states_internal1,
                        next_Zp1,
                    ) = self.model1.rnn.cell.applyRecursionOnMany(
                        latest_states_internal[:, :n1_internal], latest_Yp
                    )
                    next_states1 = np.array(next_states1)
                    next_states_internal1 = np.array(next_states_internal1)
                    next_Zp1 = np.array(next_Zp1)
                    next_Yp1 = self.model1_Cy.predict(next_states1.T).T
                    next_states[:, : self.n1] = np.array(next_states1)
                    next_states_internal[:, :n1_internal] = np.array(
                        next_states_internal1
                    )
                    next_Yp += next_Yp1
                    next_Zp += next_Zp1
                if self.n2 > 0:
                    if self.n1 > 0:
                        prior_Yp = next_Yp1
                    elif self.model2.has_prior_pred:
                        if self.model2.out_dist != "poisson":
                            noop_val_for_dist = 0  # noop for 'add'
                        else:
                            noop_val_for_dist = 1  # noop for 'multiply'
                        prior_Yp = np.ones_like(next_Yp) * noop_val_for_dist
                    else:
                        prior_Yp = None
                    (
                        next_states2,
                        next_states_internal2,
                        next_Yp2,
                    ) = self.model2.rnn.cell.applyRecursionOnMany(
                        latest_states_internal[:, n1_internal:],
                        np.concatenate(
                            (latest_states[:, : self.n1], latest_Yp), axis=1
                        ),
                        input_at_t_prior_pred=prior_Yp,
                    )
                    next_states[:, self.n1 :] = np.array(next_states2)
                    next_states_internal[:, n1_internal:] = np.array(
                        next_states_internal2
                    )
                    next_Yp = np.array(next_Yp2)
                    prior_ZpT = next_Zp1.T if self.n1 > 0 else None
                    if self.model2_Cz is not None and not self.model2_Cz_Full:
                        next_Zp = self.model2_Cz.predict(
                            next_states[:, self.n1 :].T, prior_pred=prior_ZpT
                        ).T
                if self.model1_FCy is not None and self.model1_Cy_Full:
                    next_Yp = np.array(self.model1_FCy.predict(next_states.T).T)
                if self.model2_Cz is not None and self.model2_Cz_Full:
                    next_Zp = np.array(self.model2_Cz.predict(next_states.T).T)
            allXp_all_steps.append(next_states)
            allYp_all_steps.append(next_Yp.T)
            allZp_all_steps.append(next_Zp.T)
            latest_states = np.array(next_states)
            latest_states_internal = np.array(next_states_internal)
            latest_Yp = np.array(next_Yp)
            latest_Zp = np.array(next_Zp)
            if noise_scale > 0 and self.YErrCov is not None:
                eY, eYShaping = genRandomGaussianNoise(latest_Yp.shape[0], self.YErrCov)
                latest_Yp += noise_scale * eY
        allXp_steps = [allXp_all_steps[step_ahead - 1] for step_ahead in steps_ahead]
        allYp_steps = [allYp_all_steps[step_ahead - 1] for step_ahead in steps_ahead]
        allZp_steps = [allZp_all_steps[step_ahead - 1] for step_ahead in steps_ahead]
        return allXp_steps, allYp_steps, allZp_steps

    def export_comp_graph_repr(self, savepath=None):
        out = f"{self.print_summary()}"
        if savepath is not None:
            with open(savepath, "w") as file:
                print(out, file=file)
                logger.info(f"Saved model printout as {savepath}")
        return out

    def plot_comp_graph(self, savepath="model_graph", saveExtensions=None):
        model_names = [
            "model1",
            "model2",
            "model1_Cy",
            "model1_Cy_fw",
            "model1_FCy",
            "model1_FCy_fw",
            "model2_Cz",
            "model2_Cz_fw",
        ]
        for model_name in model_names:
            model = getattr(self, model_name)
            if model is not None and hasattr(model, "plot_comp_graph"):
                model.plot_comp_graph(savepath + "_" + model_name, saveExtensions)

    def generateRealization(
        self, N, random_x0=False, x0=None, u=None, eY=None, eZ=None
    ):
        """Generates a random realization based on the DPADModel
        The model is as follows.
        x1(k+1) = A1( x1(k) ) + K1( y(k), u(k) )
        x2(k+1) = A2( x2(k) ) + K2( x1(k+1), y(k), u(k) )
        y(k)    = Cy( x1(k), x2(k), u(k) ) + ey_k
        z(k)    = Cz( x1(k), x2(k), u(k) ) + ez_k
        # Special case of linear model:
        [x1(k+1); x2(k+1)] = [A11 0; A21 A22] * [x1(k); x2(k)] + [B1 ;  B2] * [u(k); x2(k)] + [K1 ;  K2] y(y)
                      y(k) =      [Cy1   Cy2] * [x1(k); x2(k)] + [Dy1; Dy2] * [u(k); u(k)] + v(k)
                      z(k) =      [Cz1     0] * [x1(k); x2(k)] + [Dz1; Dz2] * [u(k); u(k)] + e(k)


        To generate a realization, first we draw random ey_k and ez_k, the we repeat the following given an inital x0 and any given u_k:
        z(k)    = Cz( x1(k), x2(k), u(k) ) + ez_k
        y(k)    = Cy( x1(k), x2(k), u(k) ) + ey_k
        After computing y(k) we feed it to recursion to get the next x(k+1) and then repeat from the top.
        x1(k+1) = A1( x1(k) ) + K1( y(k), u(k) )
        x2(k+1) = A2( x2(k) ) + K2( x1(k+1), y(k), u(k) )

        Args:
            N (np.array): Number of samples to generate.
            random_x0 (bool, optional): if True, will randomize the
                initial state instead of using 0 as the initial state.
                Defaults to False.
            x0 (np.array, optional): initial state. Defaults to None.
            u (np.array, optional): secondary input time series. Defaults to None.
            eY (np.array, optional): primary signal's noise time series. Defaults to None.
            eZ (np.array, optional): secondary signal's noise time series. Defaults to None.

        Returns:
            Z (np.array): secondary output time series
            Y (np.array): primary output time series
            X (np.array): latent state time series
        """
        if random_x0 and hasattr(self, "XCov") and not np.any(np.isnan(self.XCov)):
            x0 = np.atleast_2d(
                np.random.multivariate_normal(mean=np.zeros(self.nx), cov=self.XCov)
            ).T
        X = np.empty((N, self.nx))
        X[:] = np.NaN
        Y = np.empty((N, self.ny))
        Y[:] = np.NaN
        Z = np.empty((N, self.nz))
        Z[:] = np.NaN
        if eY is None:
            eY, eYShaping = genRandomGaussianNoise(N, self.YErrCov)
        if eZ is None:
            if not hasattr(self, "ZErrCov"):
                self.ZErrCov = np.zeros((self.nz, self.nz))
            eZ, eZShaping = genRandomGaussianNoise(N, self.ZErrCov)
        allXp1 = np.empty((N, self.n1))
        allXp2 = np.empty((N, self.n2))
        Xp1 = np.zeros((self.n1, 1))
        Xp2 = np.zeros((self.n2, 1))
        if x0 is not None:
            Xp1 = x0[: self.n1, :]
            Xp2 = x0[self.n1 :, :]
        if self.nu > 0 and u is None:
            u = np.zeros((N, self.nu))

        for i in range(N):
            yk_res = eY[i, :][:, np.newaxis]
            zk_res = eZ[i, :][:, np.newaxis]
            xk = np.concatenate((Xp1, Xp2), axis=0)
            if self.nu > 0:
                uk = u[i, :][:, np.newaxis]
                xk_uk = np.concatenate((Xp1, Xp2, uk), axis=0)
                x1k_uk = np.concatenate((Xp1, uk), axis=0)
                x2k_uk = np.concatenate((Xp2, uk), axis=0)
            else:
                xk_uk = xk
                x1k_uk = Xp1
                x2k_uk = Xp2

            if self.model1_Cy_Full:
                yk_res += self.model1_FCy.predict(xk_uk)
            else:
                if self.n1 > 0:
                    yk_res += self.model1_Cy.predict(x1k_uk)
                if self.n2 > 0:
                    yk_res += self.model2.rnn.cell.C.predict(x2k_uk)
            if self.model2_Cz_Full:
                zk_res += self.model2_Cz.predict(xk_uk)
            else:
                if self.n1 > 0:
                    zk_res += self.model1.rnn.cell.C.predict(x1k_uk)
                if self.n2 > 0:
                    zk_res += self.model2_Cz.predict(x2k_uk)
            yk = yk_res
            zk = zk_res

            X[i, :] = np.squeeze(xk)
            Y[i, :] = np.squeeze(yk)
            Z[i, :] = np.squeeze(zk)

            yy = Y[i, :][:, np.newaxis]
            if self.nu > 0:
                yk_uk = np.concatenate((yy, uk), axis=0)
                x1_yk_uk = np.concatenate((Xp1, yy, uk), axis=0)
                ftk = uk
            else:
                yk_uk = yy
                x1_yk_uk = np.concatenate((Xp1, yy), axis=0)
                ftk = None

            if self.n1 > 0:
                Xp1_cur, zk_next1, Xp1_next = self.model1.predict(
                    yk_uk, initial_state=Xp1.T, FT_in=ftk
                )
            else:
                Xp1_next = np.zeros(self.n1)
            Xp1_next = Xp1_next[:, np.newaxis]

            if self.n2 > 0:
                Xp2_cur, yk_next1, Xp2_next = self.model2.predict(
                    x1_yk_uk, initial_state=Xp2.T, n1_in=Xp1_next, FT_in=ftk
                )
                Xp2_next = Xp2_next[:, np.newaxis]

            if self.n1 > 0:
                Xp1 = Xp1_next
            if self.n2 > 0:
                Xp2 = Xp2_next

        return Z, Y, X
