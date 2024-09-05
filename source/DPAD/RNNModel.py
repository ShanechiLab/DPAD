""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""RNNModel class, which implements a flexible RNN"""
"""For mathematical description see RNNModelDoc.md"""


import copy
import io
import logging
import os
import re
import time
import warnings
from datetime import datetime

import tensorflow as tf

try:
    from keras.src import backend
    from keras.src.layers.rnn.rnn_utils import generate_zero_filled_state_for_cell
    from keras.src.utils import generic_utils
    from tensorflow import keras

except Exception:
    # For older tf versions
    from tensorflow.python.keras.layers.recurrent import (
        _generate_zero_filled_state_for_cell as generate_zero_filled_state_for_cell,
    )

import matplotlib.pyplot as plt
import numpy as np

from .RegressionModel import RegressionModel
from .tools.model_base_classes import ModelWithFitWithRetry, Reconstructable
from .tools.plot import plotPredictionScatter, plotTimeSeriesPrediction
from .tools.tf_losses import (
    masked_CategoricalCrossentropy,
    masked_CC,
    masked_mse,
    masked_PoissonLL_loss,
    masked_R2,
)
from .tools.tf_tools import set_global_tf_eagerly_flag
from .tools.tools import get_one_hot

logger = logging.getLogger(__name__)

make_list = lambda w: w if isinstance(w, list) else [w]


class MinimalRNNCell(tf.keras.layers.Layer):
    """A basic reference implementation of an RNN cell that implements a linear Kalman-like filter"""

    def __init__(self, units, ny_out=None, learn_initial_state=False, **kwargs):
        self.units = units
        self.state_size = units
        self.ny_out = ny_out
        self.learn_initial_state = learn_initial_state
        super(MinimalRNNCell, self).__init__(**kwargs)

    def get_config(self):
        config = super(MinimalRNNCell, self).get_config()
        initArgNames = ["units", "ny_out"]
        for fName in initArgNames:
            config[fName] = getattr(self, fName)
        return config

    def build(self, input_shape=None):
        ny_in = input_shape[0][-1]
        if self.ny_out is None:
            self.ny_out = ny_in
        self.A = self.add_weight(  # Transpose of A' or A-K*Cy
            shape=(self.units, self.units), initializer="uniform", name="A"
        )
        self.K = self.add_weight(  # Transpose of B' or K
            shape=(ny_in, self.units), initializer="uniform", name="K"
        )
        self.C = self.add_weight(
            shape=(self.units, self.ny_out),  # Transpose of Cy
            initializer="uniform",
            name="C",
        )
        self.initial_state = tf.Variable(
            name="initial_state",
            trainable=self.learn_initial_state,
            shape=(self.units,),
            initial_value=np.zeros(self.units),
            dtype=tf.float32,
        )
        self.built = True

    def call(self, input_at_t, states_at_t):
        ny_in_at_t = tf.identity(input_at_t[0], "Yk")
        prev_states = tf.identity(states_at_t[0], name="Xk")
        K_y = tf.matmul(ny_in_at_t, self.K, name="K_Yk")
        A_Xk = tf.matmul(prev_states, self.A, name="A_Xk")
        states_at_t_plus_1 = tf.add(K_y, A_Xk, name="K_Yk_plus_A_Xk")
        outs = self.propagate_steps_ahead(
            states_at_t_plus_1, states_at_t_plus_1, None, None, None
        )
        return outs, [states_at_t_plus_1]

    def propagate_steps_ahead(
        self,
        states_at_t_plus_1,
        states_at_t_plus_1_forC,
        n1_in_at_t_steps,
        ft_at_t_plus_1,
        input_at_t_prior_pred_steps,
    ):
        output_at_t = tf.matmul(
            states_at_t_plus_1_forC, self.C, name="Xk_plus_1"
        )  # Output is the prediction for t+1
        return [output_at_t, states_at_t_plus_1, states_at_t_plus_1]

    def set_cell_weights(self, weights, skip_missing=False):
        if isinstance(weights, (list, tuple)):
            self.A.assign(make_list(weights[0])[0])
            self.K.assign(make_list(weights[1])[0])
            self.C.assign(make_list(weights[2])[0])
            if len(weights) > 3:
                self.initial_state.assign(weights[3])
        else:
            self.A.assign(make_list(weights["A"])[0])
            self.C.assign(make_list(weights["C"])[0])
            if "initial_state" in weights:
                self.initial_state.assign(weights["initial_state"])
            if "K" in weights and (not skip_missing or hasattr(self, "K")):
                self.K.assign(make_list(weights["K"])[0])

    def get_cell_weights(self):
        weights = {
            "A": self.A.numpy(),
            "K": self.K.numpy(),
            "C": self.C.numpy(),
            "initial_state": self.initial_state.numpy(),
        }
        return weights

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self.initial_state is None:  # Not expected
            initial_states_for_batches = generate_zero_filled_state_for_cell(
                self, inputs, batch_size, dtype
            )
        else:
            initial_states_for_batches = tf.repeat(
                tf.expand_dims(self.initial_state, axis=0), batch_size, axis=0
            )
        return initial_states_for_batches


class NLRegRNNCell(tf.keras.layers.Layer):
    """A general RNNCell that implements each of the following 3 key mappings
       as a RegressionModel (i.e., a multilayer perceptron):
       - (1) A: mapping from the latent state from one time step to the next
       - (2) K: mapping from input to the latent state
       - (3) C: mapping from latent state to output

    If all mappings are linear, the cell will reduce to a Kalman-like filter.
    """

    def __init__(
        self,
        units,
        ny_out=None,
        nft=0,
        n1_in=0,
        ASettings={},
        KSettings={},
        CSettings={},
        unifiedAK=False,
        learn_initial_state=False,
        steps_ahead=None,  # List of ints specifying steps ahead to return from the RNN. If None will take as [1]. Default is None.
        enable_forward_pred=False,  # If True, will create a separate set of parameters (Afs, Kfw, Cfw) as needed for forward pred
        multi_step_with_A_KC=False,  # If False, will use a separate Afw(x) recursion for forward prediction, otherwise will use the regular A(x)
        use_feedthrough_in_fw=False,  # If True, will pass feedthrough even for multi-step fw prediction
        name="RNN_",
        missing_marker=None,
        **kwargs,
    ):
        self.units = units
        self.state_size = units
        self.ny_out = ny_out
        self.nft = nft  # Dimension of additional feedthrough term
        self.n1_in = n1_in  # Dimensions of inputs that are never unavailable
        self.name_prefix = name
        self.ASettings = ASettings
        self.KSettings = KSettings
        self.CSettings = CSettings
        self.unifiedAK = unifiedAK
        self.learn_initial_state = learn_initial_state
        self.steps_ahead = steps_ahead
        self.enable_forward_pred = enable_forward_pred
        if (
            isinstance(self.steps_ahead, (list, tuple))
            and np.any(np.array(self.steps_ahead) != 1)
            and self.unifiedAK
            and not self.enable_forward_pred
        ):
            # raise(Exception('When multi-step ahead prediction is of interest, for unifiedAK=True, enable_forward_pred must be True'))
            logger.warning(
                f"unifiedAK=True with enable_forward_pred=False replaces missing inputs with 0s which is technically wrong. THIS IS ALLOWED JUST FOR TESTING."
            )
        self.multi_step_with_A_KC = multi_step_with_A_KC
        self.use_feedthrough_in_fw = use_feedthrough_in_fw
        if (
            self.steps_ahead is not None
            and (len(self.steps_ahead) != 1 or self.steps_ahead[0] != 1)
            and not self.enable_forward_pred
            and not self.multi_step_with_A_KC
        ):
            raise (
                Exception(
                    "enable_forward_pred must be True to enable multi-step prediction (and to enable using a separate recursion for missing samples)"
                )
            )
        self.missing_marker = missing_marker
        # Default regression settings
        ASettingsDefaults = {
            "name": "{}A_".format(self.name_prefix),
            "units": (),
            "use_bias": False,
            "kernel_initializer": "uniform",
            "activation": "linear",
            "output_activation": "linear",
        }
        KSettingsDefaults = {
            "name": "{}K_".format(self.name_prefix),
            "units": (),
            "use_bias": False,
            "kernel_initializer": "uniform",
            "activation": "linear",
            "output_activation": "linear",
        }
        CSettingsDefaults = {
            "name": "{}C_".format(self.name_prefix),
            "units": (),
            "use_bias": False,
            "kernel_initializer": "uniform",
            "activation": "linear",
            "output_activation": "linear",
        }
        for f, v in ASettingsDefaults.items():
            if f not in self.ASettings:
                self.ASettings[f] = v
        for f, v in KSettingsDefaults.items():
            if f not in self.KSettings:
                self.KSettings[f] = v
        for f, v in CSettingsDefaults.items():
            if f not in self.CSettings:
                self.CSettings[f] = v
        if self.enable_forward_pred:
            self.AfwSettings = copy.copy(self.ASettings)
            self.AfwSettings["name"] = self.ASettings["name"].replace("_A_", "_Afw_")
            self.KfwSettings = copy.copy(self.KSettings)
            self.KfwSettings["name"] = self.KSettings["name"].replace("_K_", "_Kfw_")
            self.CfwSettings = copy.copy(self.CSettings)
            self.CfwSettings["name"] = self.CSettings["name"].replace("_C_", "_Cfw_")
        super(NLRegRNNCell, self).__init__(**kwargs)

    def get_config(self):
        config = super(NLRegRNNCell, self).get_config()
        initArgNames = [
            "units",
            "ny_out",
            "nft",
            "n1_in",
            "ASettings",
            "KSettings",
            "CSettings",
            "unifiedAK",
            "learn_initial_state",
            "steps_ahead",
            "enable_forward_pred",
            "multi_step_with_A_KC",
            "use_feedthrough_in_fw",
            "missing_marker",
        ]
        for fName in initArgNames:
            config[fName] = getattr(self, fName)
        config.update({"name": self.name_prefix})
        return config

    def build(self, input_shape=None):
        ny_in = input_shape[0][-1]
        if self.ny_out is None:
            self.ny_out = ny_in
        n_in_A = self.units if not self.unifiedAK else self.units + ny_in
        self.A = RegressionModel(n_in=n_in_A, n_out=self.units, **self.ASettings)
        if not self.unifiedAK:
            self.K = RegressionModel(n_in=ny_in, n_out=self.units, **self.KSettings)

        if self.nft > 0:
            C_in_dim = self.units + self.nft  # Also has feedthrough term
        else:
            C_in_dim = self.units
        self.C = RegressionModel(n_in=C_in_dim, n_out=self.ny_out, **self.CSettings)
        if self.enable_forward_pred:
            n_in_Afw = (
                self.units
                if not self.unifiedAK or self.n1_in == 0
                else self.units + self.n1_in
            )
            self.Afw = RegressionModel(
                n_in=n_in_Afw, n_out=self.units, **self.AfwSettings
            )  # A for recursions without input/with missing data
            if self.n1_in > 0 and not self.unifiedAK:
                self.Kfw = RegressionModel(
                    n_in=self.n1_in, n_out=self.units, **self.KfwSettings
                )  # K for inputs with missing data (only the first n1 dims available)
            if self.nft > 0:
                self.Cfw = RegressionModel(
                    n_in=self.units, n_out=self.ny_out, **self.CfwSettings
                )  # C for readouts without feedthrough
        self.initial_state = tf.Variable(
            name="initial_state",
            trainable=self.learn_initial_state,
            shape=(self.units,),
            initial_value=np.zeros(self.units),
            dtype=tf.float32,
        )
        self.built = True

    def setTrainableParameters(self, base=None, fw=None, initial_state=None):
        if base is not None:
            self.A.trainable = base
            if hasattr(self, "K"):
                self.K.trainable = base
            self.C.trainable = base
        if fw is not None:
            self.Afw.trainable = fw
            if hasattr(self, "Kfw"):
                self.Kfw.trainable = fw
            if hasattr(self, "Cfw"):
                self.Cfw.trainable = fw
        if initial_state is not None:
            self.initial_state.trainable = initial_state

    def set_cell_weights(self, weights, skip_missing=False):
        if isinstance(weights, (list, tuple)):
            self.A.set_weights(make_list(weights[0]))
            self.K.set_weights(make_list(weights[1]))
            self.C.set_weights(make_list(weights[2]))
            if len(weights) > 3:
                self.initial_state.assign(weights[3])
        else:
            self.A.set_weights(make_list(weights["A"]))
            self.C.set_weights(make_list(weights["C"]))
            if "initial_state" in weights:
                self.initial_state.assign(weights["initial_state"])
            if "K" in weights and (not skip_missing or hasattr(self, "K")):
                self.K.set_weights(make_list(weights["K"]))
            if "Afw" in weights and (not skip_missing or hasattr(self, "Afw")):
                self.Afw.set_weights(make_list(weights["Afw"]))
            if "Kfw" in weights and (not skip_missing or hasattr(self, "Kfw")):
                self.Kfw.set_weights(make_list(weights["Kfw"]))
            if "Cfw" in weights and (not skip_missing or hasattr(self, "Cfw")):
                self.Cfw.set_weights(make_list(weights["Cfw"]))

    def get_cell_weights(self):
        weights = {
            "A": self.A.get_weights(),
            "C": self.C.get_weights(),
            "initial_state": self.initial_state.numpy(),
        }
        if hasattr(self, "K"):
            weights["K"] = self.K.get_weights()
        if hasattr(self, "Afw"):
            weights["Afw"] = self.Afw.get_weights()
        if hasattr(self, "Kfw"):
            weights["Kfw"] = self.Kfw.get_weights()
        if hasattr(self, "Cfw"):
            weights["Cfw"] = self.Cfw.get_weights()
        return weights

    def call(self, input_at_t, states_at_t):
        """_summary_

        Args:
            input_at_t (list of inputs to the cell): in orders includes the following:
                - Yk
                - Uk_plus_1 (if self.nft > 0): feedthrough
                - n1_in_k (if self.n1_in): extra inputs for forecasting
                - Prior_k (if self.C.has_prior_pred): prior predictions

            states_at_t (list of previous states): Only first item matters, which should be the previous states

        Returns:
            _type_: _description_
        """
        ny_in_at_t = tf.identity(input_at_t[0], "Yk")
        if self.nft > 0:
            ft_at_t_plus_1 = tf.identity(input_at_t[1], "Uk_plus_1")
        else:
            ft_at_t_plus_1 = None
        if self.n1_in > 0:
            n1_in_at_t_steps = tf.identity(
                input_at_t[2] if self.nft > 0 else input_at_t[1], name="n1_in_k"
            )
        else:
            n1_in_at_t_steps = None
        if self.C.has_prior_pred:
            input_at_t_prior_pred_steps = tf.identity(input_at_t[-1], name="Prior_k")
        else:
            input_at_t_prior_pred_steps = None
        prev_states = tf.identity(states_at_t[0], name="Xk")
        if self.missing_marker is not None:
            mask_value_cast = tf.constant(
                self.missing_marker, dtype=ny_in_at_t.dtype, name="missing_marker"
            )
            isOk = tf.not_equal(ny_in_at_t, mask_value_cast)
            hasInput = tf.cast(
                tf.math.reduce_all(isOk, axis=1),
                dtype=ny_in_at_t.dtype,
                name="missing_mask",
            )  # Batch x 1, will be 1 for batches that have input
        else:
            hasInput = tf.ones_like(
                ny_in_at_t, dtype=ny_in_at_t.dtype, name="missing_mask"
            )[
                :, 0
            ]  # Batch x 1
        hasInputExpand = tf.expand_dims(hasInput, axis=1, name="not_missing")
        noInputExpand = tf.identity(1 - hasInputExpand, name="missing")
        # The case of non-missing data:
        states_at_t_plus_1, states_at_t_plus_1_forC = self.applyRecursion(
            prev_states, ny_in_at_t
        )
        states_at_t_plus_1_non_missing = tf.multiply(
            hasInputExpand, states_at_t_plus_1, name="K_Yk_plus_A_Xk_not_missing"
        )
        states_at_t_plus_1_forC_non_missing = tf.multiply(
            hasInputExpand,
            states_at_t_plus_1_forC,
            name="K_Yk_plus_A_Xk_forC_not_missing",
        )
        # The case of missing data:
        states_at_t_plus_1_m, states_at_t_plus_1_forC_m = self.applyForwardRecursion(
            prev_states, n1_in_at_t_steps[..., 0] if self.n1_in > 0 else None
        )
        states_at_t_plus_1_missing = tf.multiply(
            noInputExpand, states_at_t_plus_1_m, name="Xk_plus_1_missing"
        )
        states_at_t_plus_1_forC_missing = tf.multiply(
            noInputExpand, states_at_t_plus_1_forC_m, name="Xk_plus_1_forC_missing"
        )
        # Final next state
        states_at_t_plus_1 = tf.add(
            states_at_t_plus_1_non_missing, states_at_t_plus_1_missing, name="Xk_plus_1"
        )
        states_at_t_plus_1_forC = tf.add(
            states_at_t_plus_1_forC_non_missing,
            states_at_t_plus_1_forC_missing,
            name="Xk_plus_1_forC",
        )
        outs = self.propagate_steps_ahead(
            states_at_t_plus_1,
            states_at_t_plus_1_forC,
            n1_in_at_t_steps,
            ft_at_t_plus_1,
            input_at_t_prior_pred_steps,
        )
        return outs, [states_at_t_plus_1]

    def applyRecursion(self, this_state, ny_in):
        """Applies the normal recursion of states for the RNN, given the states at at time t and
        inputs at the same time
        This is used for recursions when observation is not missing

        Args:
            this_state (_type_): state at some time t
            ny_in (_type_): inputs at the same time t

        Returns:
            states_at_t_plus_1: states at the next time step
            states_at_t_plus_1: another copy of the states at the next time step to pass to predictors (useful for LSTM)
        """
        if not self.unifiedAK:  # Apply separate additive A and K
            AXk = tf.identity(
                self.A.apply_func(this_state, name_scope="A"), name="A_Xk"
            )  # Recursion
            Ky = tf.identity(
                self.K.apply_func(ny_in, name_scope="K"), name="K_Yk"
            )  # Input
            states_at_t_plus_1 = tf.add(AXk, Ky, name="K_Yk_plus_A_Xk")  # Next state
        else:  # Apply unified A,K
            XY = tf.keras.layers.concatenate(
                [this_state, ny_in], axis=-1, name="Cat_Xk_Yk"
            )
            states_at_t_plus_1 = tf.identity(
                self.A.apply_func(XY, name_scope="unifiedAK"), name="AK_Cat_Xk_Yk"
            )
        return states_at_t_plus_1, states_at_t_plus_1

    def applyForwardRecursion(self, this_state, n1_in=None):
        """Applies the forward prediction recursion of states for the RNN, given the states at at time t and
        special inputs (states from a prior stage) at the same time
        This is used for missing samples and for forecasting

        Args:
            this_state (_type_): state at some time t
            n1_in (_type_): special inputs (e.g. states from a prior stage) at the same time t

        Returns:
            states_at_t_plus_1: states at the next time step
            states_at_t_plus_1: another copy of the states at the next time step to pass to predictors (useful for LSTM)
        """
        if self.enable_forward_pred and not self.multi_step_with_A_KC:
            # self.Afw in our predictor form graph can implement/learn a separate A (not A-KC)
            if not self.unifiedAK or n1_in is None:
                next_states = tf.identity(
                    self.Afw.apply_func(this_state, name_scope="Afw"), name="Afw_Xk"
                )
                if n1_in is not None:
                    next_states += tf.identity(
                        self.Kfw.apply_func(n1_in, name_scope="Kfw"), name="Kfw_n1_in"
                    )
            else:  # Apply unified Afw,Kfw
                XN1 = tf.keras.layers.concatenate(
                    [this_state, n1_in], axis=-1, name="Cat_Xk_n1_in"
                )
                next_states = tf.identity(
                    self.Afw.apply_func(XN1, name_scope="unifiedAfwKfw"),
                    name="AfwKfw_Cat_Xk_n1_in",
                )
        else:  # For older models
            # self.A in our predictor form graph actually implements/learns A-KC, or in the case of self.unifiedAK, it learns (A-KC)(x)+K(y)
            if not self.unifiedAK:
                next_states = tf.identity(
                    self.A.apply_func(this_state, name_scope="A"), name="A_Xk"
                )  # Recursion
            else:
                # raise(Exception('Network does not support missing samples. Set enable_forward_pred=True.'))
                # TEMP, to enable loading older models that didn't have enable_forward_pred:
                # Technically not correct, just for testing:
                Y0 = tf.zeros(
                    tuple(this_state.shape[:-1])
                    + (self.A.n_in - this_state.shape[-1],),
                    name="Y0",
                )
                XY = tf.keras.layers.concatenate(
                    [this_state, Y0], axis=-1, name=f"Cat_Xk_Y0"
                )
                next_states = tf.identity(
                    self.A.apply_func(XY, name_scope="unifiedAK"), name="AK_Cat_Xk_Y0"
                )
        return next_states, next_states

    def get_pred_from_states_and_feedthrough(
        self, states_at_t_plus_1_forC, ft_at_t_plus_1, input_at_t_prior_pred
    ):
        C_mapping = self.C
        if self.nft > 0 and ft_at_t_plus_1 is not None:
            C_in_at_t_plus_1 = tf.keras.layers.concatenate(
                [states_at_t_plus_1_forC, ft_at_t_plus_1], axis=-1, name="Cat_Xk_Uk"
            )
        else:
            C_in_at_t_plus_1 = states_at_t_plus_1_forC
            if self.nft > 0:
                C_mapping = (
                    self.Cfw
                )  # Generally can have feedthrough, but we are not supposed to use it in this prediction either because it is multiple steps ahead, or just unavailable
        if self.C.has_prior_pred and input_at_t_prior_pred is not None:
            output_at_t = C_mapping.apply_func(
                [C_in_at_t_plus_1, input_at_t_prior_pred], name_scope="C_with_prior"
            )  # Output is the prediction for t+1
        else:
            output_at_t = C_mapping.apply_func(
                C_in_at_t_plus_1, name_scope="C"
            )  # Output is the prediction for t+1
        return output_at_t

    def propagate_steps_ahead(
        self,
        states_at_t_plus_1,
        states_at_t_plus_1_forC,
        n1_in_at_t_steps,
        ft_at_t_plus_1,
        input_at_t_prior_pred_steps,
    ):
        if self.steps_ahead is None:
            output_at_t = self.get_pred_from_states_and_feedthrough(
                states_at_t_plus_1_forC,
                ft_at_t_plus_1,
                (
                    None
                    if input_at_t_prior_pred_steps is None
                    else input_at_t_prior_pred_steps[..., 0]
                ),
            )
            outs = [output_at_t, states_at_t_plus_1_forC, states_at_t_plus_1]
        else:
            steps_ahead_preds = {}
            steps_ahead_states = {}
            steps_ahead_internal_states = {}
            maxStepAhead = np.max(self.steps_ahead)
            minStepAhead = 1 if 0 not in self.steps_ahead else 0
            for saInd, step_ahead in enumerate(range(minStepAhead, 1 + maxStepAhead)):
                with tf.name_scope(f"{step_ahead}-step") as scope:
                    # Propagate the states
                    if step_ahead == minStepAhead:
                        next_states, next_states_forC = (
                            states_at_t_plus_1,
                            states_at_t_plus_1_forC,
                        )  # The input states must be for 1-step ahead
                    else:
                        # this_n1 = n1_in_at_t_steps[..., saInd-1] if hasattr(self, 'n1_in') and self.n1_in > 0 else None
                        # print('n1 being passed to applyForwardRecursion: {}'.format(this_n1))
                        next_states, next_states_forC = self.applyForwardRecursion(
                            this_state,
                            (
                                n1_in_at_t_steps[..., saInd - 1]
                                if hasattr(self, "n1_in") and self.n1_in > 0
                                else None
                            ),
                        )
                    this_state = tf.identity(
                        next_states, name=f"states_at_t_plus_{step_ahead}"
                    )
                    if step_ahead in self.steps_ahead:
                        # Get the prediction from the propagated state
                        if self.nft > 0 and (
                            step_ahead == minStepAhead or self.use_feedthrough_in_fw
                        ):  # Or maybe step_ahead==1, double check!
                            ft_at_t_plus_1_this_step = ft_at_t_plus_1
                        else:
                            ft_at_t_plus_1_this_step = None
                        if (
                            self.C.has_prior_pred
                            and input_at_t_prior_pred_steps is not None
                        ):
                            this_step_ind = [
                                saInd
                                for saInd, s in enumerate(self.steps_ahead)
                                if s == step_ahead
                            ]  # Find the index corresponding to this step-ahead
                            input_at_t_prior_pred_this_step = (
                                input_at_t_prior_pred_steps[..., this_step_ind[0]]
                            )
                        else:
                            input_at_t_prior_pred_this_step = None
                        next_pred = self.get_pred_from_states_and_feedthrough(
                            next_states_forC,
                            ft_at_t_plus_1_this_step,
                            input_at_t_prior_pred_this_step,
                        )
                        this_pred = tf.identity(
                            next_pred,
                            name=(
                                f"output_at_t_plus_{step_ahead-1}"
                                if step_ahead > 0
                                else f"output_at_t_minus_1"
                            ),
                        )
                        steps_ahead_states[step_ahead] = (
                            next_states_forC  # Actually used for predictions
                        )
                        steps_ahead_internal_states[step_ahead] = (
                            next_states  # True underlying state of RNN/LSTM (same as above for RNN)
                        )
                        steps_ahead_preds[step_ahead] = this_pred
            outs = []
            for step_ahead in self.steps_ahead:
                outs.append(steps_ahead_preds[step_ahead])
            for step_ahead in self.steps_ahead:
                outs.append(steps_ahead_states[step_ahead])
            for step_ahead in self.steps_ahead:
                outs.append(steps_ahead_internal_states[step_ahead])
        return outs

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self.initial_state is None:  # Not expected
            initial_states_for_batches = generate_zero_filled_state_for_cell(
                self, inputs, batch_size, dtype
            )
        else:
            initial_states_for_batches = tf.repeat(
                tf.expand_dims(self.initial_state, axis=0), batch_size, axis=0
            )
        return initial_states_for_batches

    def applyRecursionOnMany(
        self, this_state, ny_in, ft_at_t_plus_1=None, input_at_t_prior_pred=None
    ):
        states_at_t_plus_1, states_at_t_plus_1_forC = self.applyRecursion(
            this_state, ny_in
        )
        output_at_t = self.get_pred_from_states_and_feedthrough(
            states_at_t_plus_1_forC, ft_at_t_plus_1, input_at_t_prior_pred
        )
        return states_at_t_plus_1_forC, states_at_t_plus_1, output_at_t


class NLRegLSTMCell(NLRegRNNCell):
    """A special case of NLRegRNNCell that replaces the A mapping with an LSTM cell."""

    def __init__(
        self,
        units,
        ny_out=None,
        nft=0,
        n1_in=0,
        ASettings={},
        KSettings={},
        CSettings={},
        unifiedAK=False,
        learn_initial_state=False,
        LSTMSettings={},
        steps_ahead=None,  # List of ints specifying steps ahead to return from the RNN. If None will take as [1]. Default is None.
        enable_forward_pred=False,  # If True, will create a separate set of parameters (Afs, Kfw, Cfw) as needed for forward pred
        multi_step_with_A_KC=False,  # If False, will use a separate Afw(x) recursion for forward prediction, otherwise will use the regular A(x)
        use_feedthrough_in_fw=False,  # If True, will pass feedthrough even for multi-step fw prediction
        name="LSTM_",
        missing_marker=None,
        **kwargs,
    ):
        self.units = units
        self.state_size = 2 * units
        self.ny_out = ny_out
        self.nft = nft  # Dimension of additional feedthrough term
        self.n1_in = n1_in  # Dimensions of inputs that are never unavailable
        self.name_prefix = name
        self.LSTMSettings = LSTMSettings
        self.ASettings = ASettings
        self.KSettings = KSettings
        self.CSettings = CSettings
        self.unifiedAK = unifiedAK
        self.learn_initial_state = learn_initial_state
        self.steps_ahead = steps_ahead
        self.enable_forward_pred = enable_forward_pred
        if (
            isinstance(self.steps_ahead, (list, tuple))
            and np.any(np.array(self.steps_ahead) != 1)
            and self.unifiedAK
            and not self.enable_forward_pred
        ):
            # raise(Exception('When multi-step ahead prediction is of interest, for unifiedAK=True, enable_forward_pred must be True'))
            logger.warning(
                f"unifiedAK=True with enable_forward_pred=False replaces missing inputs with 0s which is technically wrong. THIS IS ALLOWED JUST FOR TESTING."
            )
        self.multi_step_with_A_KC = multi_step_with_A_KC
        self.use_feedthrough_in_fw = use_feedthrough_in_fw
        if (
            self.steps_ahead is not None
            and (len(self.steps_ahead) != 1 or self.steps_ahead[0] != 1)
            and not self.enable_forward_pred
            and not self.multi_step_with_A_KC
        ):
            raise (
                Exception(
                    "enable_forward_pred must be True to enable multi-step prediction (and to enable using a separate recursion for missing samples)"
                )
            )
        self.missing_marker = missing_marker
        # Default LSTM settings
        LSTMSettingsDefaults = {
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
            "unit_forget_bias": True,
            "kernel_regularizer": None,
            "recurrent_regularizer": None,
            "bias_regularizer": None,
            "kernel_constraint": None,
            "recurrent_constraint": None,
            "bias_constraint": None,
            "dropout": 0.0,
            "recurrent_dropout": 0.0,
            "name": "A_LSTM",
        }
        for f, v in LSTMSettingsDefaults.items():
            if f not in self.LSTMSettings:
                self.LSTMSettings[f] = v
        # Default regression settings
        ASettingsDefaults = {
            "name": "{}A_".format(self.name_prefix),
            "units": (),
            "use_bias": False,
            "kernel_initializer": "uniform",
            "activation": "linear",
            "output_activation": "linear",
        }
        KSettingsDefaults = {
            "name": "{}K_".format(self.name_prefix),
            "units": (),
            "use_bias": False,
            "kernel_initializer": "uniform",
            "activation": "linear",
            "output_activation": "linear",
        }
        CSettingsDefaults = {
            "name": "{}C_".format(self.name_prefix),
            "units": (),
            "use_bias": False,
            "kernel_initializer": "uniform",
            "activation": "linear",
            "output_activation": "linear",
        }
        for f, v in ASettingsDefaults.items():
            if f not in self.ASettings:
                self.ASettings[f] = v
        for f, v in KSettingsDefaults.items():
            if f not in self.KSettings:
                self.KSettings[f] = v
        for f, v in CSettingsDefaults.items():
            if f not in self.CSettings:
                self.CSettings[f] = v
        if self.enable_forward_pred:
            self.LSTMfwSettings = copy.copy(self.LSTMSettings)
            self.LSTMfwSettings["name"] = self.LSTMSettings["name"].replace(
                "_A_LSTM", "_Afw_LSTM"
            )
            self.KfwSettings = copy.copy(self.KSettings)
            self.KfwSettings["name"] = self.KSettings["name"].replace("_K_", "_Kfw_")
            self.CfwSettings = copy.copy(self.CSettings)
            self.CfwSettings["name"] = self.CSettings["name"].replace("_C_", "_Cfw_")
        super(NLRegRNNCell, self).__init__(**kwargs)

    def get_config(self):
        config = super(NLRegLSTMCell, self).get_config()
        initArgNames = [
            "units",
            "ny_out",
            "nft",
            "n1_in",
            "ASettings",
            "KSettings",
            "CSettings",
            "unifiedAK",
            "learn_initial_state",
            "LSTMSettings",
            "steps_ahead",
            "enable_forward_pred",
            "multi_step_with_A_KC",
            "use_feedthrough_in_fw",
            "missing_marker",
        ]
        for fName in initArgNames:
            config[fName] = getattr(self, fName)
        config.update({"name": self.name_prefix})
        return config

    def build(self, input_shape=None):
        ny_in = input_shape[0][-1]
        if self.ny_out is None:
            self.ny_out = ny_in
        # n_in_A = self.units if not self.unifiedAK else self.units+ny_in  # LSTMCell will deduce input dimension automatically
        self.LSTMCell = tf.keras.layers.LSTMCell(units=self.units, **self.LSTMSettings)
        if not self.unifiedAK:
            self.K = RegressionModel(n_in=ny_in, n_out=self.units, **self.KSettings)

        if self.nft > 0:
            C_in_dim = self.units + self.nft  # Also has feedthrough term
        else:
            C_in_dim = self.units
        self.C = RegressionModel(n_in=C_in_dim, n_out=self.ny_out, **self.CSettings)
        if self.enable_forward_pred:
            # n_in_Afw = self.units  # LSTMCell will deduce input dimension automatically
            self.LSTMCellfw = tf.keras.layers.LSTMCell(
                units=self.units, **self.LSTMfwSettings
            )  # A for recursions without input/with missing data
            if self.n1_in > 0:
                self.Kfw = RegressionModel(
                    n_in=self.n1_in, n_out=self.units, **self.KfwSettings
                )  # K for inputs with missing data (only the first n1 dims available)
            if self.nft > 0:
                self.Cfw = RegressionModel(
                    n_in=self.units, n_out=self.ny_out, **self.CfwSettings
                )  # C for readouts without feedthrough
        self.initial_state = tf.Variable(
            name="initial_state",
            trainable=self.learn_initial_state,
            shape=(self.units * 2,),
            initial_value=np.zeros(self.units * 2),
            dtype=tf.float32,
        )
        self.built = True

    def setTrainableParameters(self, base=None, fw=None, initial_state=None):
        if base is not None:
            self.LSTMCell.trainable = base
            if hasattr(self, "K"):
                self.K.trainable = base
            self.C.trainable = base
        if fw is not None:
            self.LSTMCellfw.trainable = fw
            if hasattr(self, "Kfw"):
                self.Kfw.trainable = fw
            if hasattr(self, "Cfw"):
                self.Cfw.trainable = fw
        if initial_state is not None:
            self.initial_state.trainable = initial_state

    def set_cell_weights(self, weights, skip_missing=False):
        if isinstance(weights, (list, tuple)):
            self.LSTMCell.set_weights(make_list(weights[0]))
            self.K.set_weights(make_list(weights[1]))
            self.C.set_weights(make_list(weights[2]))
            if len(weights) > 3:
                self.initial_state.assign(weights[3])
        else:
            self.LSTMCell.set_weights(make_list(weights["LSTMCell"]))
            self.C.set_weights(make_list(weights["C"]))
            if "initial_state" in weights:
                self.initial_state.assign(weights["initial_state"])
            if "K" in weights and (not skip_missing or hasattr(self, "K")):
                self.K.set_weights(make_list(weights["K"]))
            if "LSTMCellfw" in weights and (
                not skip_missing or hasattr(self, "LSTMCellfw")
            ):
                self.LSTMCellfw.set_weights(make_list(weights["LSTMCellfw"]))
            if "Kfw" in weights and (not skip_missing or hasattr(self, "Kfw")):
                self.Kfw.set_weights(make_list(weights["Kfw"]))
            if "Cfw" in weights and (not skip_missing or hasattr(self, "Cfw")):
                self.Cfw.set_weights(make_list(weights["Cfw"]))

    def get_cell_weights(self):
        weights = {
            "LSTMCell": self.LSTMCell.get_weights(),
            "C": self.C.get_weights(),
            "initial_state": self.initial_state.numpy(),
        }
        if hasattr(self, "K"):
            weights["K"] = self.K.get_weights()
        if hasattr(self, "LSTMCellfw"):
            weights["LSTMCellfw"] = self.LSTMCellfw.get_weights()
        if hasattr(self, "Kfw"):
            weights["Kfw"] = self.Kfw.get_weights()
        if hasattr(self, "Cfw"):
            weights["Cfw"] = self.Cfw.get_weights()
        return weights

    def applyRecursion(self, this_state, ny_in):
        """Applies the normal recursion of states for the RNN, given the states at at time t and
        inputs at the same time
        This is used for recursions when observation is not missing

        Args:
            this_state (_type_): state at some time t
            ny_in (_type_): inputs at the same time t

        Returns:
            states_at_t_plus_1: states at the next time step
        """
        prev_states_mem = tf.identity(
            this_state[:, : self.units], name="Xk_mem"
        )  # previous memory state
        prev_states_carry = tf.identity(
            this_state[:, self.units :], name="Xk_carry"
        )  # previous carry state
        states_at_t_all = (prev_states_mem, prev_states_carry)
        if not self.unifiedAK:  # Apply separate additive A and K
            LSTM_input_at_t = tf.identity(
                self.K.apply_func(ny_in, name_scope="K"), name="K_Yk"
            )  # Input
        else:  # Apply unified A,K
            LSTM_input_at_t = ny_in
        LSTM_new_output, LSTM_new_states = self.LSTMCell(
            LSTM_input_at_t, states_at_t_all
        )
        states_at_t_plus_1 = tf.concat(
            LSTM_new_states, axis=-1, name="Xk_plus_1_LSTMCat"
        )
        return states_at_t_plus_1, LSTM_new_output

    def applyForwardRecursion(self, this_state, n1_in=None):
        """Applies the forward prediction recursion of states for the RNN, given the states at at time t and
        special inputs (states from a prior stage) at the same time
        This is used for missing samples and for forecasting

        Args:
            this_state (_type_): state at some time t
            n1_in (_type_): special inputs (e.g. states from a prior stage) at the same time t

        Returns:
            states_at_t_plus_1: states at the next time step
        """
        prev_states_mem = tf.identity(
            this_state[:, : self.units], name="Xk_mem"
        )  # previous memory state
        prev_states_carry = tf.identity(
            this_state[:, self.units :], name="Xk_carry"
        )  # previous carry state
        states_at_t_all = (prev_states_mem, prev_states_carry)

        if self.enable_forward_pred and not self.multi_step_with_A_KC:
            if n1_in is None:
                LSTMfw_input_at_t = tf.zeros(
                    tuple(this_state.shape[:-1]) + (1,), name="Y0"
                )  # No input for forecasting
            elif not self.unifiedAK:
                LSTMfw_input_at_t = self.Kfw.apply_func(n1_in, name_scope="Kfw")
            else:
                LSTMfw_input_at_t = n1_in
            LSTM_new_output, LSTM_new_states = self.LSTMCellfw(
                LSTMfw_input_at_t, states_at_t_all
            )  # Recursion forward prediction
            next_states = tf.concat(
                LSTM_new_states, axis=-1, name="Xk_plus_1_LSTMfwCat"
            )
        else:  # For older models
            # self.A in our predictor form graph actually implements/learns A-KC, or in the case of self.unifiedAK, it learns (A-KC)(x)+K(y)
            # Technically not correct, just for testing:
            if not self.unifiedAK:
                Y0 = tf.zeros(tuple(this_state.shape[:-1]) + (self.K.n_in,), name="Y0")
                LSTM_input_at_t = tf.identity(
                    self.K.apply_func(Y0, name_scope="K"), name="K_Y0"
                )  # Input
            else:
                LSTM_n_in = self.LSTMCell.kernel.n_in
                LSTM_input_at_t = tf.zeros(
                    tuple(this_state.shape[:-1]) + (LSTM_n_in,), name="Y0"
                )
            LSTM_new_output, LSTM_new_states = self.LSTMCell(
                LSTM_input_at_t, states_at_t_all
            )
            next_states = tf.concat(LSTM_new_states, axis=-1, name="Xk_plus_1_LSTMCat")
        return next_states, LSTM_new_output


class Bidirectional(tf.keras.layers.Bidirectional):
    def call(
        self,
        inputs,
        training=None,
        mask=None,
        initial_state=None,
        constants=None,
    ):
        """`Bidirectional.call` implements the same API as the wrapped `RNN`."""
        kwargs = {}
        if generic_utils.has_arg(self.layer.call, "training"):
            kwargs["training"] = training
        if generic_utils.has_arg(self.layer.call, "mask"):
            kwargs["mask"] = mask
        if generic_utils.has_arg(self.layer.call, "constants"):
            kwargs["constants"] = constants

        if generic_utils.has_arg(self.layer.call, "initial_state"):
            if isinstance(inputs, list) and len(inputs) > 1:
                # initial_states are keras tensors, which means they are passed
                # in together with inputs as list. The initial_states need to be
                # split into forward and backward section, and be feed to layers
                # accordingly.
                forward_inputs = [inputs[0]]
                backward_inputs = [inputs[0]]
                pivot = (len(inputs) - self._num_constants) // 2 + 1
                # add forward initial state
                forward_inputs += inputs[1:pivot]
                if not self._num_constants:
                    # add backward initial state
                    backward_inputs += inputs[pivot:]
                else:
                    # add backward initial state
                    backward_inputs += inputs[pivot : -self._num_constants]
                    # add constants for forward and backward layers
                    forward_inputs += inputs[-self._num_constants :]
                    backward_inputs += inputs[-self._num_constants :]
                forward_state, backward_state = None, None
                if "constants" in kwargs:
                    kwargs["constants"] = None
            elif initial_state is not None:
                # initial_states are not keras tensors, eg eager tensor from np
                # array.  They are only passed in from kwarg initial_state, and
                # should be passed to forward/backward layer via kwarg
                # initial_state as well.
                forward_inputs, backward_inputs = inputs, inputs
                half = len(initial_state) // 2
                forward_state = initial_state[:half]
                backward_state = initial_state[half:]
            else:
                forward_inputs, backward_inputs = inputs, inputs
                forward_state, backward_state = None, None

            y = self.forward_layer(
                forward_inputs, initial_state=forward_state, **kwargs
            )
            y_rev = self.backward_layer(
                backward_inputs, initial_state=backward_state, **kwargs
            )
        else:
            y = self.forward_layer(inputs, **kwargs)
            y_rev = self.backward_layer(inputs, **kwargs)

        if self.return_state:
            states = y[1:] + y_rev[1:]
            y = y[0]
            y_rev = y_rev[0]

        # Custom modifications to support cells that return state *sequence* as the second output (e.g., NLRegRNNCell)
        def prep_output(y, y_rev, merge_mode):
            if self.return_sequences:
                time_dim = 0 if getattr(self.forward_layer, "time_major", False) else 1
                y_rev = backend.reverse(y_rev, time_dim)
            if merge_mode == "concat":
                output = backend.concatenate([y, y_rev])
            elif merge_mode == "sum":
                output = y + y_rev
            elif merge_mode == "ave":
                output = (y + y_rev) / 2
            elif merge_mode == "mul":
                output = y * y_rev
            elif merge_mode is None:
                output = [y, y_rev]
            else:
                raise ValueError(
                    "Unrecognized value for `merge_mode`. "
                    f"Received: {merge_mode}"
                    'Expected values are ["concat", "sum", "ave", "mul"]'
                )
            return output

        if isinstance(y, (list)):
            output = []
            for yi, (y_this, y_rev_this) in enumerate(zip(y, y_rev)):
                output_this = prep_output(
                    y_this,
                    y_rev_this,
                    merge_mode="concat" if yi == 1 else self.merge_mode,
                )  # For the state, always concatenate
                output.append(output_this)
        else:
            output = prep_output(y, y_rev)

        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return [output] + states
        return output


class RNNModel(ModelWithFitWithRetry, Reconstructable):
    """An RNN model that uses NLRegRNNCell to enable fine grained control over the nonlinearity of each element of the RNN."""

    def __init__(
        self,
        nx: int | None = None,
        ny: int | None = None,
        block_samples: int | None = None,
        batch_size: int | None = None,
        ny_out: int | None = None,
        nft: int = 0,
        n1_in: int = 0,
        out_dist: (
            str | None
        ) = None,  # Output distribution, default: None which means 'gaussian', can also be 'poisson'
        has_prior_pred: bool = False,  # If true, will support a prior prediction during training
        name: str = "RNN_",  # Name in computation graph
        log_dir: str = "",  # If not empty, will save logs with tensorboard
        linear_cell: bool = False,
        LSTM_cell: bool = False,
        missing_marker: float | None = None,
        cell_args: dict = {},
        bidirectional: bool = False,  # If true, will duplicate the RNN to have a backwards pass
        bidir_merge_mode: (
            str | None
        ) = None,  # How to combine the outputs of the two directions (default: None). If None, determine based on distribution
        initLSSM=None,  # An LSSM to initialize the RNN with
        initLSSM_backward=None,  # An LSSM to initialize the backward RNN with
        optimizer_name="Adam",  # Name or constructor of optimizer
        optimizer_args: dict | None = None,  # Dict of arguments for the optimizer
        lr_scheduler_name=None,  # The name of learning rate scheduler to use, e.g., ExponentialDecay
        lr_scheduler_args: (
            dict | None
        ) = None,  # The arguments of the learning rate scheduler to use
        stateful: bool = True,
        steps_ahead: (
            list[int] | None
        ) = None,  # Number of steps ahead in terms states and predictions that the RNN should return. Can be a list of ints. If None, will treat as [1]
        steps_ahead_loss_weights: (
            list[float] | None
        ) = None,  # Weight of each step ahead prediction in loss. If None, will give all steps ahead equal weight of 1.
        enable_forward_pred: bool = False,  # If true will enable forward prediction and handling of missing inputs by adding a separate set of Afw,Kfw,Cfw parameters
        multi_step_with_A_KC: bool = False,  # If False, will use a separate Afw(x) for forward prediction, otherwise will use the same A(x)
        use_feedthrough_in_fw: bool = False,  # If True, will pass feedthrough even for multi-step fw prediction
    ):
        self.constructor_kwargs = {
            "nx": nx,
            "ny": ny,
            "block_samples": block_samples,
            "batch_size": batch_size,
            "ny_out": ny_out,
            "nft": nft,
            "n1_in": n1_in,
            "out_dist": out_dist,
            "has_prior_pred": has_prior_pred,
            "name": name,
            "log_dir": log_dir,
            "linear_cell": linear_cell,
            "LSTM_cell": LSTM_cell,
            "missing_marker": missing_marker,
            "cell_args": cell_args,
            "bidirectional": bidirectional,
            "bidir_merge_mode": bidir_merge_mode,
            "initLSSM": initLSSM,
            "initLSSM_backward": initLSSM_backward,
            "optimizer_name": optimizer_name,
            "optimizer_args": optimizer_args,
            "lr_scheduler_name": lr_scheduler_name,
            "lr_scheduler_args": lr_scheduler_args,
            "stateful": stateful,
            "steps_ahead": steps_ahead,
            "steps_ahead_loss_weights": steps_ahead_loss_weights,
            "enable_forward_pred": enable_forward_pred,
            "multi_step_with_A_KC": multi_step_with_A_KC,
            "use_feedthrough_in_fw": use_feedthrough_in_fw,
        }
        if initLSSM is not None:
            nx = initLSSM.state_dim
            ny = initLSSM.output_dim + initLSSM.input_dim
            ny_out = initLSSM.output_dim
            nft = initLSSM.input_dim
            out_dist = None
            cell_args = {}
            if initLSSM_backward is not None:
                if initLSSM_backward.output_dim != ny_out:
                    raise (
                        Exception(
                            f"Forward and backward models must have the same number of output dimensions"
                        )
                    )

        self.nx: int = nx
        self.ny: int = ny
        if ny_out is None:
            ny_out = ny
        self.ny_out: int = ny_out
        self.nft: int = nft
        self.n1_in = n1_in
        self.out_dist = out_dist  # Default: 'normal', can also be 'poisson'

        if self.out_dist == "poisson":
            if (
                "CSettings" in cell_args
                and "out_dist" in cell_args["CSettings"]
                and cell_args["CSettings"]["out_dist"] != self.out_dist
            ):
                logger.warning(
                    'Overwriting "out_dist" for RNN\'s output projection to be "{}" (from "{}")'.format(
                        self.out_dist, cell_args["CSettings"]["out_dist"]
                    )
                )
            cell_args["CSettings"]["out_dist"] = self.out_dist

        self.has_prior_pred = has_prior_pred
        if self.has_prior_pred:
            cell_args["CSettings"]["has_prior_pred"] = self.has_prior_pred

        if "KSettings" in cell_args and "unifiedAK" in cell_args["KSettings"]:
            cell_args["unifiedAK"] = cell_args["KSettings"]["unifiedAK"]
            del cell_args["KSettings"]["unifiedAK"]

        self.block_samples = block_samples
        self.batch_size = batch_size
        self.cell_args = cell_args
        self.missing_marker = missing_marker

        self.name = name
        self.log_dir = log_dir
        self.log_subdir = ""
        self.bidirectional = bidirectional
        self.bidir_merge_mode = bidir_merge_mode
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_args = (
            lr_scheduler_args if lr_scheduler_args is not None else {}
        )
        self.stateful = stateful
        self.steps_ahead = steps_ahead
        self.steps_ahead_loss_weights = steps_ahead_loss_weights
        self.enable_forward_pred = enable_forward_pred
        self.multi_step_with_A_KC = multi_step_with_A_KC
        self.use_feedthrough_in_fw = use_feedthrough_in_fw

        self.linear_cell = linear_cell
        self.LSTM_cell = LSTM_cell
        self.build()
        if initLSSM is not None:
            if not self.bidirectional:
                self.setToLSSM(initLSSM)
            else:
                self.setToLSSM(initLSSM, cell=self.rnn.forward_layer.cell)
                self.setToLSSM(initLSSM_backward, cell=self.rnn.backward_layer.cell)
            self.set_batch_size(
                batch_size=1
            )  # Switch to batch size of 1 to make prediction exact

    def get_config(self):
        config = super(RNNModel, self).get_config()
        initArgNames = [
            "nx",
            "ny",
            "block_samples",
            "batch_size",
            "ny_out",
            "nft",
            "out_dist",
            "n1_in",
            "has_prior_pred",
            "log_dir",
            "linear_cell",
            "LSTM_cell",
            "missing_marker",
            "cell_args",
            "bidirectional",
            "bidir_merge_mode",
            "initLSSM",
            "optimizer_name",
            "optimizer_args",
            "lr_scheduler_name",
            "lr_scheduler_args",
            "stateful",
            "steps_ahead",
            "steps_ahead_loss_weights",
            "enable_forward_pred",
            "multi_step_with_A_KC",
            "use_feedthrough_in_fw",
            "name",
        ]
        for fName in initArgNames:
            config[fName] = getattr(self, fName)
        return config

    def build(self):
        self.steps_ahead = (
            self.steps_ahead
            if self.steps_ahead is not None
            else ([1] if not self.bidirectional else [0])
        )
        y_in = tf.keras.Input(
            shape=(self.block_samples, self.ny), batch_size=self.batch_size
        )  # y_k is the input to the RNN
        if self.linear_cell:
            if "ASettings" in self.cell_args:
                self.cell_args.pop("ASettings")
            if "KSettings" in self.cell_args:
                self.cell_args.pop("KSettings")
            if "CSettings" in self.cell_args:
                self.cell_args.pop("CSettings")
            cell = MinimalRNNCell(self.nx, ny_out=self.ny_out, **self.cell_args)
        else:
            NLRegRNNCellClass = NLRegLSTMCell if self.LSTM_cell else NLRegRNNCell
            cell = NLRegRNNCellClass(
                self.nx,
                ny_out=self.ny_out,
                nft=self.nft,
                n1_in=self.n1_in,
                missing_marker=self.missing_marker,
                steps_ahead=self.steps_ahead,
                enable_forward_pred=self.enable_forward_pred,
                multi_step_with_A_KC=self.multi_step_with_A_KC,
                use_feedthrough_in_fw=self.use_feedthrough_in_fw,
                name=self.name,
                **self.cell_args,
            )
        num_steps_ahead = 1 if self.steps_ahead is None else len(self.steps_ahead)
        max_steps_ahead = (
            1 if self.steps_ahead is None else int(np.max(self.steps_ahead + [1]))
        )
        y_in = (y_in,)
        if self.nft > 0:
            feedthrough_input = tf.keras.Input(
                shape=(
                    self.block_samples,
                    self.nft,
                ),
                batch_size=self.batch_size,
                name="feedthrough_input",
            )
            y_in += (feedthrough_input,)
        if self.n1_in > 0:
            # We need fw pred input (x from stage 1) for all steps ahead
            fw_pred_y_in = tf.keras.Input(
                shape=(
                    self.block_samples,
                    self.n1_in,
                    max_steps_ahead,
                ),
                batch_size=self.batch_size,
                name="rnn_fw_pred_inputs",
            )  # this is the input to the RNN used for forward prediction (e.g. states from a prior stage)
            y_in += (fw_pred_y_in,)
        if self.has_prior_pred:
            prior_pred = tf.keras.Input(
                shape=(
                    self.block_samples,
                    self.ny_out,
                    num_steps_ahead,
                ),
                batch_size=self.batch_size,
                name="rnn_prior_pred",
            )
            y_in += (prior_pred,)
        self.y_in = y_in
        if not self.bidirectional:
            self.rnn = tf.keras.layers.RNN(
                cell, return_sequences=True, stateful=self.stateful
            )
        else:
            self.rnn_fw = tf.keras.layers.RNN(
                cell, return_sequences=True, stateful=self.stateful
            )
            self.rnn_bw = tf.keras.layers.RNN(
                cell, return_sequences=True, stateful=self.stateful, go_backwards=True
            )
            if self.bidir_merge_mode is None:
                if self.out_dist != "poisson":
                    self.bidir_merge_mode = "sum"
                else:
                    self.bidir_merge_mode = "mul"
            self.rnn = Bidirectional(
                self.rnn_fw,
                backward_layer=self.rnn_bw,
                merge_mode=self.bidir_merge_mode,
            )
        self.rnn_outs = self.rnn(self.y_in)  # y_k+1|k is the output
        self.y_out = self.rnn_outs[0]
        if self.steps_ahead is None:
            self.x_hat = self.rnn_outs[1]
        else:
            self.x_hat = self.rnn_outs[len(self.steps_ahead)]

        self.model = tf.keras.models.Model(inputs=self.y_in, outputs=self.rnn_outs)

        if self.steps_ahead is not None:
            self.model.output_names = (
                [f"{self.rnn.name}_{step}step" for step in self.steps_ahead]
                + [f"{self.rnn.name}_{step}step_state" for step in self.steps_ahead]
                + [
                    f"{self.rnn.name}_{step}step_internal_state"
                    for step in self.steps_ahead
                ]
            )

        self.compile()

    def compile(self):
        metrics = []
        if self.isOutputCategorical():
            if self.missing_marker is None:
                loss = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True
                )  # Later will need softmax for pred_model
            else:
                loss = masked_CategoricalCrossentropy(
                    self.missing_marker
                )  # Later will need softmax for pred_model
        elif self.out_dist == "poisson":
            if self.missing_marker is None:
                loss = "poisson"
            else:
                loss = masked_PoissonLL_loss(self.missing_marker)
        else:
            if self.missing_marker is None:
                loss = "mse"  # Without masking
                # loss = tf.keras.losses.MeanSquaredError()
            else:
                loss = masked_mse(self.missing_marker)
            metrics.append(masked_R2(self.missing_marker))
            metrics.append(masked_CC(self.missing_marker))
        metrics.append(loss)

        if isinstance(self.lr_scheduler_name, str):
            if hasattr(tf.keras.optimizers.schedules, self.lr_scheduler_name):
                lr_scheduler_constructor = getattr(
                    tf.keras.optimizers.schedules, self.lr_scheduler_name
                )
            else:
                raise Exception(
                    "Learning rate scheduler {self.lr_scheduler_name} not supported as string, pass actual class for the optimizer (e.g. tf.keras.optimizers.Adam)"
                )
        else:
            lr_scheduler_constructor = self.lr_scheduler_name
        if isinstance(self.optimizer_name, str):
            if (
                self.optimizer_name == "adam"
            ):  # For backward compatibility with old saved models
                self.optimizer_name = "Adam"
            if hasattr(tf.keras.optimizers, self.optimizer_name):
                optimizer_constructor = getattr(
                    tf.keras.optimizers, self.optimizer_name
                )
            else:
                raise Exception(
                    "optimizer not supported as string, pass actual class for the optimizer (e.g. tf.keras.optimizers.Adam)"
                )
        else:
            optimizer_constructor = self.optimizer_name
        if lr_scheduler_constructor is not None:
            if (
                "learning_rate" in self.optimizer_args
                and "initial_learning_rate" not in self.lr_scheduler_args
            ):
                self.lr_scheduler_args["initial_learning_rate"] = self.optimizer_args[
                    "learning_rate"
                ]
            lr_scheduler = lr_scheduler_constructor(**self.lr_scheduler_args)
            self.optimizer_args["learning_rate"] = lr_scheduler
        optimizer = optimizer_constructor(**self.optimizer_args)
        if self.steps_ahead is None:
            all_losses = [loss, None, None]
            loss_weights = None
        else:
            all_losses = (
                [loss] * len(self.steps_ahead)
                + [None] * len(self.steps_ahead)
                + [None] * len(self.steps_ahead)
            )
            metrics = (
                [metrics] * len(self.steps_ahead)
                + [None] * len(self.steps_ahead)
                + [None] * len(self.steps_ahead)
            )  # For each output
        if self.steps_ahead_loss_weights is not None:
            loss_weights = (
                [float(lw) for lw in self.steps_ahead_loss_weights]
                + [0] * len(self.steps_ahead)
                + [0] * len(self.steps_ahead)
            )  # For each output
        else:
            loss_weights = None
        self.model.compile(
            optimizer=optimizer,
            loss=all_losses,
            loss_weights=loss_weights,
            metrics=metrics,
        )
        self.model.run_eagerly = False  # You can temporarily set this to True to enable eager execution so that you can put breakpoints inside the model for debugging. But eager execution will be EXTREMELY slow.

    def isOutputCategorical(self):
        return (
            "CSettings" in self.cell_args
            and "num_classes" in self.cell_args["CSettings"]
            and self.cell_args["CSettings"]["num_classes"] is not None
        )

    def fit(
        self,
        Y_in,
        Y_out,
        FT_in=None,
        n1_in=None,
        prior_pred=None,
        Y_in_val=None,
        Y_out_val=None,
        FT_in_val=None,
        n1_in_val=None,
        prior_pred_val=None,
        epochs=100,
        verbose=0,
        prediction_batch_size=1,
        init_attempts=1,  # Number of initialization retries for each model fitting attempt. Will keep the best outcome after each series of attempts
        max_attempts=1,  # Maximum number of times that the whole model fitting will be repeated in case of a blow-up or nan loss
        throw_on_fail=False,  # If true, will throw an error if the final learned model is unstable
        prior_pred_shift_by_one=False,  # Will be passes to self.predict after fitting only when checking for blow-ups
        shift_preds=None,  # If True, will shift outputs by one to train model for one step ahead prediction as in:
        #     ===> index i has seen inputs up to BUT NOT including index i (so index 0 has seen no input)
        # If False, will not shift the RNN outputs, so the predictions will be as follows:
        #     ===> index i has seen inputs up to AND including index i (useful for bidirectional)
        # If None, will be set to True iff self.bidirectional is True
        early_stopping_patience=3,
        start_from_epoch=0,
        early_stopping_measure="loss",
    ):

        if shift_preds is None:
            shift_preds = not self.bidirectional
        steps_ahead = (
            [1 if shift_preds else 0] if self.steps_ahead is None else self.steps_ahead
        )
        max_steps_ahead = np.max(steps_ahead)
        shift_samples = 1 if shift_preds else 0

        logger.info(
            f"Shifting output by {shift_samples} samples to train network for {shift_samples}-step ahead prediction"
        )

        def prep_IO_data(Y_in, Y_out, FT_in, n1_in, prior_pred):
            n_input, Ndat = Y_in.shape[0], Y_in.shape[1]
            num_batch = int(
                np.floor(
                    (Ndat - max_steps_ahead) / self.block_samples / self.batch_size
                )
            )

            # Remove batches that have no yOutTrain data
            yInRemInds = []
            yOutRemInds = []
            yOutStepsRemInds = [[] for sa in steps_ahead]
            remBatchInds = []
            for bi in range(num_batch):
                yInBatchInds = np.arange(
                    (bi * (self.batch_size * self.block_samples)),
                    (bi + 1) * (self.batch_size * self.block_samples),
                )
                yOutBatchInds = np.arange(
                    shift_samples + (bi * (self.batch_size * self.block_samples)),
                    shift_samples + ((bi + 1) * (self.batch_size * self.block_samples)),
                )
                yOutStepsBatchInds = [
                    np.arange(
                        sa + (bi * (self.batch_size * self.block_samples)),
                        sa + ((bi + 1) * (self.batch_size * self.block_samples)),
                    )
                    for sa in steps_ahead
                ]
                isOkInd = np.nonzero(Y_out[:, yOutBatchInds] != self.missing_marker)[1]
                if len(isOkInd) == 0:  # Empty batch
                    remBatchInds.append(bi)
                    yInRemInds.extend(yInBatchInds)
                    yOutRemInds.extend(yOutBatchInds)
                    for saInd, sa in enumerate(steps_ahead):
                        yOutStepsRemInds[saInd].extend(yOutStepsBatchInds)

            yInInds = np.arange(0, num_batch * self.batch_size * self.block_samples)
            yOutInds = np.arange(
                shift_samples,
                shift_samples + (num_batch * self.batch_size * self.block_samples),
            )
            yOutIndsSteps = [
                np.arange(sa, sa + (num_batch * self.batch_size * self.block_samples))
                for sa in steps_ahead
            ]
            if len(yInRemInds) > 0:
                logger.warning(
                    "Discarding {} of {} batches because they have no data".format(
                        len(remBatchInds), num_batch
                    )
                )
                yInInds = yInInds[~np.isin(yInInds, yInRemInds)]
                yOutInds = yOutInds[~np.isin(yOutInds, yOutRemInds)]
                yOutIndsSteps = [
                    yOutIndsThisStep[
                        ~np.isin(yOutIndsThisStep, yOutStepsRemInds[saInd])
                    ]
                    for saInd, yOutIndsThisStep in enumerate(yOutIndsSteps)
                ]
                num_batch = num_batch - len(remBatchInds)

            if num_batch < 1:
                return None, None, num_batch

            yInTrain = Y_in[:, yInInds].T.reshape(
                (num_batch * self.batch_size, self.block_samples, self.ny)
            )
            yOutTrain = Y_out[:, yOutInds].T.reshape(
                (num_batch * self.batch_size, self.block_samples, self.ny_out)
            )
            yOutStepsTrain = [
                Y_out[:, yOutIndsThisStep].T.reshape(
                    (num_batch * self.batch_size, self.block_samples, self.ny_out)
                )
                for yOutIndsThisStep in yOutIndsSteps
            ]

            inputTrain = (yInTrain,)
            if self.nft > 0:
                # Feedthrough is from the same indices as the output
                ftTrain = FT_in[:, yOutInds].T.reshape(
                    (num_batch * self.batch_size, self.block_samples, self.nft)
                )
                inputTrain += (ftTrain,)
            elif FT_in is not None and FT_in.size:
                raise (
                    Exception(
                        "Feedthrough data is given, but is not expected in the model"
                    )
                )

            if self.isOutputCategorical():
                yOutTrain = get_one_hot(
                    np.array(yOutTrain, dtype=int), self.y_out.shape[-1]
                )
                yOutStepsTrain = [
                    get_one_hot(
                        np.array(yOutThisStepTrain, dtype=int), self.y_out.shape[-1]
                    )
                    for yOutThisStepTrain in yOutStepsTrain
                ]
            if self.n1_in > 0:
                n1InTrainList = [
                    n1_in_step[:, yOutInds].T.reshape(
                        (num_batch * self.batch_size, self.block_samples, self.n1_in)
                    )
                    for n1_in_step in n1_in
                ]
                n1InTrain = np.moveaxis(np.array(n1InTrainList), 0, -1)
                inputTrain += (n1InTrain,)
            if self.has_prior_pred:
                if prior_pred is not None:
                    yOutStepsTrainPriorList = [
                        prior_pred_this[:, yOutInds].T.reshape(
                            (
                                num_batch * self.batch_size,
                                self.block_samples,
                                self.ny_out,
                            )
                        )
                        for prior_pred_this in prior_pred
                    ]
                else:
                    if self.out_dist != "poisson":
                        yOutStepsTrainPriorList = [
                            np.zeros_like(yOutThisStepTrain)
                            for yOutThisStepTrain in yOutStepsTrain
                        ]  # noop for 'add'
                    else:
                        yOutStepsTrainPriorList = [
                            np.ones_like(yOutThisStepTrain)
                            for yOutThisStepTrain in yOutStepsTrain
                        ]  # noop for 'multiply'
                yOutStepsTrainPrior = np.moveaxis(
                    np.array(yOutStepsTrainPriorList), 0, -1
                )
                inputTrain += (yOutStepsTrainPrior,)

            dummy_x_steps = [
                np.zeros((yOutTrainStep.shape[0], yOutTrainStep.shape[1], self.nx))
                for yOutTrainStep in yOutStepsTrain
            ]
            nx_internal = 2 * self.nx if self.LSTM_cell else self.nx
            dummy_x_internal_steps = [
                np.zeros((yOutTrainStep.shape[0], yOutTrainStep.shape[1], nx_internal))
                for yOutTrainStep in yOutStepsTrain
            ]
            yOutStepsTrain = (
                yOutStepsTrain + dummy_x_steps + dummy_x_internal_steps
            )  # To have outputs as lists
            return inputTrain, yOutStepsTrain, num_batch

        yInTrain, yOutTrain, num_batch = prep_IO_data(
            Y_in, Y_out, FT_in, n1_in, prior_pred
        )
        logger.info(
            "Have {} batches each with {} {}-sample data segments (ny_in={}, ny_out={}, nft={})".format(
                num_batch,
                self.batch_size,
                self.block_samples,
                self.ny,
                self.ny_out,
                self.nft,
            )
        )

        validation_data = None
        if Y_in_val is not None:
            validation_input, validation_output, num_batch_val = prep_IO_data(
                Y_in_val, Y_out_val, FT_in_val, n1_in_val, prior_pred_val
            )
            if num_batch_val > 0:
                validation_data = (validation_input, validation_output)

        batch_size_backup = self.batch_size
        fitWasOk, attempt = False, 0
        while not fitWasOk and attempt < max_attempts:
            attempt += 1
            history = self.fit_with_retry(
                init_attempts=init_attempts,
                early_stopping_patience=early_stopping_patience,
                early_stopping_measure=early_stopping_measure,
                start_from_epoch=start_from_epoch,
                # The rest of the arguments will be passed to keras model.fit
                x=yInTrain,
                y=yOutTrain,
                shuffle=False,
                batch_size=self.batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=verbose,
            )
            if (
                prediction_batch_size is not None
                and prediction_batch_size != self.batch_size
            ):
                self.set_batch_size(
                    batch_size=prediction_batch_size
                )  # Switch to batch size of 1 to facilitate future predictions
            if max_attempts > 1 or throw_on_fail:
                # Check for blow-ups
                preds = self.predict(
                    Y_in,
                    FT_in=FT_in,
                    n1_in=n1_in,
                    prior_pred=prior_pred,
                    prior_pred_shift_by_one=prior_pred_shift_by_one,
                )
                allXp = preds[0]
                fitWasOk = not np.isnan(history.history["loss"][-1]) or not np.any(
                    np.isnan(allXp)
                )  # Unstable, states blow up [OR is TEMP]
                if not fitWasOk:
                    weights_backup = self.get_cell_weights()
                    nan_weights = [
                        k
                        for k, v in weights_backup.items()
                        if isinstance(v, list)
                        and np.any([np.any(np.isnan(vThis)) for vThis in v])
                    ]
                    if attempt < max_attempts:
                        logger.info(
                            f"RNN fit was not stable (blew-up and led to nan loss). Nan-weights: {nan_weights}. Retrying with attempt {attempt+1}"
                        )
                        self.set_batch_size(
                            batch_size=batch_size_backup
                        )  # Switch back to batch size needed for model fitting
                        # Reinitialize weights
                        if (
                            "optimizer_args" not in self.constructor_kwargs
                            or self.constructor_kwargs["optimizer_args"] is None
                        ):
                            self.constructor_kwargs["optimizer_args"] = {}
                        if (
                            "global_clipnorm"
                            not in self.constructor_kwargs["optimizer_args"]
                        ):
                            self.constructor_kwargs["optimizer_args"][
                                "global_clipnorm"
                            ] = 10
                        self = RNNModel(**self.constructor_kwargs)
                    else:
                        msg = f"RNN fit was not stable (blew-up and led to nan loss). Nan-weights: {nan_weights}. All {attempt} attempts have failed!"
                        logger.error(msg)
                        if throw_on_fail:
                            raise (Exception(msg))
        return history

    def set_steps_ahead(self, steps_ahead):
        weights = self.get_cell_weights()
        self.steps_ahead = steps_ahead
        self.build()
        self.set_cell_weights(weights, skip_missing=True)

    def set_multi_step_with_A_KC(self, multi_step_with_A_KC):
        weights = self.get_cell_weights()
        self.multi_step_with_A_KC = multi_step_with_A_KC
        self.build()
        self.set_cell_weights(weights)

    def set_use_feedthrough_in_fw(self, use_feedthrough_in_fw):
        self.use_feedthrough_in_fw = use_feedthrough_in_fw
        # Rebuild cell to ensure the setting takes effect in tf
        self.set_steps_ahead(self.steps_ahead)

    def get_cells(self):
        if not self.bidirectional:
            return {"fw": self.rnn.cell}
        else:
            return {
                "fw": self.rnn.forward_layer.cell,
                "bw": self.rnn.backward_layer.cell,
            }

    def set_batch_size(self, batch_size):
        weights = self.get_cell_weights()
        self.batch_size = batch_size
        self.build()
        self.set_cell_weights(weights)

    def set_cell_weights(self, weights, skip_missing=False, cell=None):
        if cell is None:
            if not self.bidirectional:
                cell = self.rnn.cell
            else:
                for name, cell in self.get_cells().items():
                    self.set_cell_weights(
                        weights[name], skip_missing=skip_missing, cell=cell
                    )
                return
        if isinstance(cell, NLRegRNNCell) or hasattr(cell, "set_cell_weights"):
            cell.set_cell_weights(weights, skip_missing=skip_missing)
        else:
            cell.set_weights(weights)

    def get_cell_weights(self, cell=None):
        if cell is None:
            if not self.bidirectional:
                cell = self.rnn.cell
            else:
                return {
                    name: self.get_cell_weights(cell)
                    for name, cell in self.get_cells().items()
                }
        if isinstance(cell, NLRegRNNCell) or hasattr(cell, "get_cell_weights"):
            weights = cell.get_cell_weights()
        else:
            weights = cell.get_weights()
        return weights

    def setTrainableParameters(self, base=None, fw=None, initial_state=None):
        for name, cell in self.get_cells().items():
            cell.setTrainableParameters(base=base, fw=fw, initial_state=initial_state)
        self.compile()  # "make sure to call compile() again on your model for your changes to be taken into account."

    def setToLSSM(self, s, cell=None):
        if s.input_dim > 0:
            w = {
                "A": [s.A_KC.T],
                "K": [np.concatenate((s.K, s.B_KD), axis=1).T],
                "C": [np.concatenate((s.C, s.D), axis=1).T],
            }
        else:
            w = {"A": [s.A_KC.T], "K": [s.K.T], "C": [s.C.T]}
        if hasattr(s, "x0") and s.x0 is not None:
            w["initial_state"] = s.x0
        else:
            w["initial_state"] = np.zeros((self.nx,))
        if self.steps_ahead is not None and (
            len(self.steps_ahead) > 1 or self.steps_ahead[0] != 1
        ):
            w["Afw"] = [s.A.T]
            if self.nft > 0:
                w["Cfw"] = [s.C.T]
        self.set_cell_weights(w, cell=cell)

    def predict_with_keras(self, inputs, initial_state=None):
        steps_ahead = [1] if self.steps_ahead is None else self.steps_ahead
        yHat = [None for saInd in range(len(steps_ahead))]

        Y_in = inputs[0]
        FT_in = inputs[1] if self.nft > 0 else None
        n1_in = (inputs[2] if self.nft > 0 else inputs[1]) if self.n1_in > 0 else None
        prior_pred = inputs[-1] if self.has_prior_pred else None
        if Y_in is not None and len(Y_in.shape) == 3:
            Y_in = np.concatenate(list(Y_in), axis=0).T
        if FT_in is not None and len(FT_in.shape) == 3:
            FT_in = np.concatenate(list(FT_in), axis=0).T
        if (
            n1_in is not None and len(n1_in.shape) == 4
        ):  # Trial-based, for each step ahead
            n1_in = np.reshape(
                n1_in, (n1_in.shape[0] * n1_in.shape[1], n1_in.shape[2], n1_in.shape[3])
            ).transpose((2, 1, 0))
        if (
            prior_pred is not None and len(prior_pred.shape) == 4
        ):  # Trial-based, for each step ahead
            prior_pred = np.reshape(
                prior_pred,
                (
                    prior_pred.shape[0] * prior_pred.shape[1],
                    prior_pred.shape[2],
                    prior_pred.shape[3],
                ),
            ).transpose((2, 1, 0))
        (
            predictor_input,
            feedthrough_res,
            FT0,
            n1_in0_steps,
            pp0_steps,
        ) = self.prepare_predict_inputs(
            Y_in, FT_in=FT_in, n1_in=n1_in, prior_pred=prior_pred
        )
        eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        if self.rnn.stateful:
            self.rnn.reset_states(
                states=initial_state
            )  # Set the initial RNN state to zero
        rnn_outs = self.model.predict(predictor_input, batch_size=self.batch_size)
        set_global_tf_eagerly_flag(eagerly_flag_backup)
        return rnn_outs
        # batch_count = inputs[0].shape[0]/self.batch_size
        # if int(batch_count) != batch_count:
        #     raise(Exception('First dimension must be a multiple of batch_size ({}). Other cases are not implemented.'.format(self.batch_size)))
        # eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        # batch_count = int(batch_count)
        # for bi in range(batch_count): # Pass in batches, because keras expects the same
        #     xThisBatch = tuple([val[(bi*self.batch_size):((bi+1)*self.batch_size), ...] for val in inputs])
        #     rnn_outs = self.model.predict(xThisBatch)
        #     for saInd in range(len(steps_ahead)):
        #         yHatThisBatch = rnn_outs[saInd]
        #         yHat[saInd] = yHatThisBatch if bi == 0 else np.concatenate((yHat[saInd], yHatThisBatch))
        #     set_global_tf_eagerly_flag(eagerly_flag_backup)
        # return tuple(yHat)

    def reshape_to_epochs_and_maybe_pop_first(
        self, in_y, data_dim, shift_by_one=False, append_val=0.0
    ):
        n_input, Ndat = in_y.shape[0], in_y.shape[1]
        num_batch = int(np.ceil(Ndat / (self.batch_size * self.block_samples)))
        output_shape = (num_batch * self.batch_size, self.block_samples, data_dim)

        if self.rnn.stateful:
            in_y_ext = np.concatenate(
                (
                    in_y,
                    append_val
                    * np.ones(
                        (
                            data_dim,
                            num_batch * self.batch_size * self.block_samples - Ndat,
                        )
                    ),
                ),
                axis=1,
            )
            first_samples = in_y_ext[:, 0][np.newaxis, :]
            if shift_by_one:
                # Ignore the first sample and append a dummy sample at the end instead (useful for feedthrough/prior)
                in_y_ext = np.concatenate(
                    (in_y_ext[:, 1:], append_val * np.ones((in_y_ext.shape[0], 1))),
                    axis=1,
                )
            in_y_ext_res = in_y_ext.T.reshape(output_shape)
        else:
            in_y_ext_res = in_y.T.reshape(output_shape)
            first_samples = in_y_ext_res[:, 0, ...]
            if shift_by_one:
                # Ignore the first sample and append a dummy sample at the end instead (useful for feedthrough/prior)
                in_y_ext_res = np.concatenate(
                    (
                        in_y_ext_res[:, 1:, ...],
                        append_val
                        * np.ones((in_y_ext_res.shape[0], 1, in_y_ext_res.shape[2])),
                    ),
                    axis=1,
                )
        return in_y_ext_res, first_samples

    def prepare_predict_inputs(
        self,
        Y_in,
        FT_in=None,
        n1_in=None,
        prior_pred=None,
        FT_in_shift_by_one=True,  # (This should be True according to the formulation's time indexing) If true, will move feedthrough time step forward (normally needed because u_t affects x_t+1, but it also affects y_t)
        prior_pred_shift_by_one=False,  # If true, will move prior_pred one time step forward (useful when prior_pred comes from stage 1)
    ):
        steps_ahead = [1] if self.steps_ahead is None else self.steps_ahead
        n_input, Ndat = Y_in.shape[0], Y_in.shape[1]
        num_batch = int(np.ceil(Ndat / (self.batch_size * self.block_samples)))

        Y_in_ext_res, y0 = self.reshape_to_epochs_and_maybe_pop_first(
            Y_in, self.ny, shift_by_one=False, append_val=0.0
        )

        predictor_input = (Y_in_ext_res,)
        if self.nft > 0:
            feedthrough_res, FT0 = self.reshape_to_epochs_and_maybe_pop_first(
                FT_in, self.nft, shift_by_one=FT_in_shift_by_one, append_val=0.0
            )
            predictor_input += (feedthrough_res,)
        else:
            feedthrough_res, FT0 = None, None

        if self.n1_in > 0:
            n1_in_steps_list, n1_in0_steps_list = [], []
            for n1_in_this in n1_in:
                (
                    n1_in_this_res,
                    n1_in0_this,
                ) = self.reshape_to_epochs_and_maybe_pop_first(
                    n1_in_this, self.n1_in, shift_by_one=True, append_val=0.0
                )
                n1_in_steps_list.append(n1_in_this_res)
                n1_in0_steps_list.append(n1_in0_this)
            n1_in_steps = np.moveaxis(np.array(n1_in_steps_list), 0, -1)
            n1_in0_steps = np.moveaxis(np.array(n1_in0_steps_list), 0, -1)
            predictor_input += (n1_in_steps,)
        else:
            n1_in0_steps = None

        if self.has_prior_pred:
            if self.out_dist != "poisson":
                noop_val_for_dist = 0  # noop for 'add'
            else:
                noop_val_for_dist = 1  # noop for 'multiply'
            if prior_pred is not None:
                prior_pred_in = [
                    (
                        pp
                        if pp is not None
                        else np.ones((self.ny_out, Ndat)) * noop_val_for_dist
                    )
                    for pp in prior_pred
                ]
            else:
                prior_pred_in = [
                    np.ones((self.ny_out, Ndat)) * noop_val_for_dist
                    for s in steps_ahead
                ]
            prior_pred_res_steps, pp0_steps = [], []
            for saInd in range(len(steps_ahead)):
                prior_pred_res, pp0 = self.reshape_to_epochs_and_maybe_pop_first(
                    prior_pred_in[saInd],
                    self.ny_out,
                    shift_by_one=prior_pred_shift_by_one,
                    append_val=noop_val_for_dist,
                )
                prior_pred_res_steps.append(prior_pred_res)
                pp0_steps.append(pp0)
            prior_pred_res_steps = np.moveaxis(np.array(prior_pred_res_steps), 0, -1)
            pp0_steps = np.moveaxis(np.array(pp0_steps), 0, -1)
            predictor_input += (prior_pred_res_steps,)
        else:
            pp0_steps = None
        return predictor_input, feedthrough_res, FT0, n1_in0_steps, pp0_steps

    def predict(
        self,
        Y_in,
        FT_in=None,
        n1_in=None,
        prior_pred=None,
        use_quick_method=None,
        initial_state=None,
        FT_in_shift_by_one=True,  # (This should be True according to the formulation's time indexing) If true, will move feedthrough time step forward (normally needed because u_t affects x_t+1, but it also affects y_t)
        prior_pred_shift_by_one=False,  # If true, will move prior_pred one time step forward (useful when prior_pred comes from stage 1)
        shift_preds=None,  # If True, will shift predictions by one, and propagate x0 through the output, to make sure the samples of
        # returned states and predictions are as follows:
        #     ===> index i has seen inputs up to BUT NOT including index i (so index 0 has seen no input)
        # If False, will not shift the RNN outputs, so the predictions will be as follows:
        #     ===> index i has seen inputs up to AND including index i (useful for bidirectional)
        # If None, will be set to True iff self.bidirectional is True
        return_internal_states=False,  # If true, will return internal RNN states (useful for LSTM)
        verbose=0,
    ):
        """Predicts the output from the input of the RNN

        Returns:
            if steps_ahead has n elements:
            - (    1) to (  n): allXp_steps =>
            - (  n+1) to (2*n): allYp_steps
            - (2*n+1) to (3*n): Xp_steps
            if return_internal_states:
            - (3*n+1) to (4*n): allXp_internal_steps
            - (4*n+1) to (5*n): Xp_internal_steps
        """
        if shift_preds is None:
            shift_preds = not self.bidirectional
        n_input, Ndat = Y_in.shape[0], Y_in.shape[1]
        if n_input != self.ny:
            raise (
                Exception(
                    "Unexpected input dimensions! Input to predict must be dim x samples with dim={}.".format(
                        self.ny
                    )
                )
            )
        if FT_in is not None and FT_in.shape[0] != self.nft:
            raise (
                Exception(
                    "Unexpected feedthrough signal dimension! Feedthrough signal must be dim x samples with dim={}.".format(
                        self.nft
                    )
                )
            )
        if n1_in is not None and self.n1_in > 0:
            if not isinstance(n1_in, (list, tuple)):
                n1_in = [n1_in]
            for saInd, n1_in_this in enumerate(n1_in):
                if n1_in_this is not None and n1_in_this.shape[0] != self.n1_in:
                    raise (
                        Exception(
                            f"Unexpected n1_in signal dimension (for the {saInd}-th step ahead)! Prior prediction signal must be dim x samples with dim={self.n1_in}."
                        )
                    )
        if prior_pred is not None:
            if not isinstance(prior_pred, (list, tuple)):
                prior_pred = [prior_pred]
            for saInd, prior_pred_this in enumerate(prior_pred):
                if (
                    prior_pred_this is not None
                    and prior_pred_this.shape[0] != self.ny_out
                ):
                    raise (
                        Exception(
                            f"Unexpected prior_pred signal dimension (for the {saInd}-th step ahead)! Prior prediction signal must be dim x samples with dim={self.ny_out}."
                        )
                    )
        if initial_state is not None and self.LSTM_cell:
            initial_state = np.concatenate((initial_state, initial_state), axis=1)
        steps_ahead = [1] if self.steps_ahead is None else self.steps_ahead
        if (
            self.batch_size == 1 and use_quick_method is not False
        ) or use_quick_method is True:
            # Faster implementation, but only exact if batch_size == 1 (it is noisy around edges of batches)
            if not self.rnn.stateful and Ndat % self.block_samples != 0:
                raise (
                    Exception(
                        "For non-stateful (trial-based RNNs), input data must be a multiple of block_samples ({})".format(
                            self.block_samples
                        )
                    )
                )
            num_batch = int(np.ceil(Ndat / (self.batch_size * self.block_samples)))

            (
                predictor_input,
                feedthrough_res,
                FT0,
                n1_in0_steps,
                pp0_steps,
            ) = self.prepare_predict_inputs(
                Y_in,
                FT_in=FT_in,
                n1_in=n1_in,
                prior_pred=prior_pred,
                FT_in_shift_by_one=FT_in_shift_by_one,
                prior_pred_shift_by_one=prior_pred_shift_by_one,
            )

            eagerly_flag_backup = set_global_tf_eagerly_flag(False)  # Temp
            if self.rnn.stateful:
                self.rnn.reset_states(
                    states=initial_state
                )  # Set the initial RNN state to zero
            rnn_outs = self.model.predict(
                predictor_input, batch_size=self.batch_size, verbose=verbose
            )  # xHat_{1|0} to xHat_{Ndat|Ndat-1} and yHat_{1|0} to yHat_{Ndat|Ndat-1}
            set_global_tf_eagerly_flag(eagerly_flag_backup)
            steps_ahead = self.steps_ahead if self.steps_ahead is not None else [1]
            y_hat_steps = rnn_outs[: len(steps_ahead)]
            x_hat_steps = rnn_outs[
                len(steps_ahead) : (2 * len(steps_ahead))
            ]  # For LSTM this is the output, used as the state in subsequent modeling
            x_hat_internal_steps = rnn_outs[
                (2 * len(steps_ahead)) : (3 * len(steps_ahead))
            ]  # For LSTM this is the internal state
            y_hat, x_hat, x_hat_internal = (
                y_hat_steps[0],
                x_hat_steps[0],
                x_hat_internal_steps[0],
            )
            if (
                not shift_preds
            ):  # Return outputs as is, without trying to have index i be the prediction before seeing input index i

                def reshape_batches(in_x, N):
                    # Expected dimensions of in_x: (epochs x time x dim) or for categorical: (epochs x time x dim x num classes)
                    new_cat_shape = tuple(
                        [num_batch * self.batch_size * self.block_samples]
                        + list(in_x.shape[2:])
                    )  # Dims: time x dim / for categorical: time x dim x num classes
                    if self.rnn.stateful:
                        in_xR = in_x.reshape(new_cat_shape)
                        allXp = in_xR[:N, ...]
                    else:
                        N_each = int(N / in_x.shape[0])
                        allXp = in_x[:, :N_each, ...].reshape(
                            new_cat_shape
                        )  # Dims: time x dim / for categorical: time x dim x num classes
                    return allXp

                Xp_steps = [
                    None for x in x_hat_steps
                ]  # Final step is not being popped out, so nothing to return
                for si in range(len(steps_ahead)):
                    # Reshape XHat
                    allXpThis = reshape_batches(x_hat_steps[si], Ndat).T
                    allXpInternalThis = reshape_batches(
                        x_hat_internal_steps[si], Ndat
                    ).T
                    allYpThis = reshape_batches(y_hat_steps[si], Ndat)
                    # Reshape YHat
                    if not self.isOutputCategorical():
                        allYpThis = allYpThis.T
                    else:
                        allYp_logits = allYpThis.transpose([1, 0, 2])
                        # Convert to class probabilities by applying softmax
                        softmax = tf.keras.layers.Softmax(
                            input_shape=allYp_logits.shape
                        )
                        allYpThis = softmax(allYp_logits).numpy()
                    x_hat_steps[si] = allXpThis
                    x_hat_internal_steps[si] = allXpInternalThis
                    y_hat_steps[si] = allYpThis
                outs = tuple(x_hat_steps) + tuple(y_hat_steps) + tuple(Xp_steps)
                if return_internal_states:
                    Xp_internal_steps = [
                        None for x in x_hat_steps
                    ]  # Final step is not being popped out, so nothing to return
                    outs += tuple(x_hat_internal_steps) + tuple(Xp_internal_steps)
                return outs
            # Try to shift predictions and pass x0 to readout so that in the final return predictions, index i has seen up to but not including input at index i
            # This process doesn't make sense for bidirectional
            cells = self.get_cells()
            if initial_state is not None:
                x0 = np.array(initial_state).flatten()
            elif np.any(
                [
                    True
                    for cell in cells.values()
                    if hasattr(cell, "initial_state") and cell.initial_state is not None
                ]
            ):
                x0 = np.concatenate(
                    [np.array(cell.initial_state).flatten() for cell in cells.values()]
                )
            else:
                x0 = np.zeros((x_hat.shape[-1],))
            if self.bidirectional:
                # The "initial_state" for the backward pass is actually the first sample of the inferred states above
                x0[self.nx : (2 * self.nx)] = x_hat[0, 0, self.nx : (2 * self.nx)]
            if self.LSTM_cell:
                x0_full = x0
                if not self.bidirectional:
                    x0 = x0[self.nx :]  # Keep the memory units (ignore the carry units)
                else:
                    x0 = np.concatenate(
                        (x0[self.nx : (2 * self.nx)], x0[(2 * self.nx) : (3 * self.nx)])
                    )  # Keep the memory units (ignore the carry units)
            else:
                x0_full = x0

            def prepend_x0_and_pop_last_sample(in_x, x0_val, N):
                # Expected dimensions of in_x: (epochs x time x dim) or for categorical: (epochs x time x dim x num classes)
                if len(x0_val.shape) == 1:
                    x0T = x0_val[np.newaxis, :]
                else:
                    x0T = x0_val
                new_cat_shape = tuple(
                    [num_batch * self.batch_size * self.block_samples]
                    + list(in_x.shape[2:])
                )  # Dims: time x dim / for categorical: time x dim x num classes
                if self.rnn.stateful:
                    in_xR = in_x.reshape(new_cat_shape)
                    in_xR = np.concatenate((x0T, in_xR), axis=0)
                    allXp = in_xR[:N, ...]
                    Xp = in_xR[N, ...]  # X(Ndat+1|Ndat)
                else:
                    N_each = int(N / in_x.shape[0])
                    if x0T.shape[0] == in_x.shape[0]:
                        x0ForAllBatches = x0T.reshape(
                            [x0T.shape[0], 1, x0T.shape[1]]
                            + (
                                []
                                if len(in_x.shape) <= 3
                                else [1] * (len(in_x.shape) - 3)
                            )
                        )
                    else:
                        x0ForAllBatches = x0T.reshape(
                            [1, 1, x0T.size]
                            + (
                                []
                                if len(in_x.shape) <= 3
                                else [1] * (len(in_x.shape) - 3)
                            )
                        ).repeat(num_batch * self.batch_size, axis=0)
                    in_xR = np.concatenate((x0ForAllBatches, in_x), axis=1)
                    allXp = in_xR[:, :N_each, ...].reshape(
                        new_cat_shape
                    )  # Dims: time x dim / for categorical: time x dim x num classes
                    Xp = in_xR[:, N_each, ...]  # X(Ndat+1|Ndat), for each batch
                return allXp, Xp

            if steps_ahead is not None:
                x0T = x0[np.newaxis, :]
                x0_fullT = x0_full[np.newaxis, :]
                if self.nft > 0 and FT_in is not None and FT_in_shift_by_one:
                    FT0T = FT0
                    if not self.rnn.stateful and (
                        FT0.shape[0] == feedthrough_res.shape[0]
                    ):  # If trial based
                        x0T = np.tile(x0T.T, [1, feedthrough_res.shape[0]]).T
                        x0_fullT = np.tile(x0_fullT.T, [1, feedthrough_res.shape[0]]).T
                else:
                    FT0T = None
                if self.n1_in > 0:
                    n1_in0 = n1_in0_steps
                else:
                    n1_in0 = None
                if self.has_prior_pred:
                    prior_pred_step1 = pp0_steps
                else:
                    prior_pred_step1 = None
                if not self.bidirectional:
                    outs = self.rnn.cell.propagate_steps_ahead(
                        x0_fullT, x0T, n1_in0, FT0T, prior_pred_step1
                    )  # Propagate x0 through the dynamics
                    steps_ahead_x0s = [
                        np.array(x)[0:1, ...]
                        for x in outs[len(steps_ahead) : (2 * len(steps_ahead))]
                    ]
                    steps_ahead_internal_x0s = [
                        np.array(x)[0:1, ...]
                        for x in outs[(2 * len(steps_ahead)) : (3 * len(steps_ahead))]
                    ]
                    steps_ahead_y_hat0 = [
                        np.array(y)[0:1, ...] for y in outs[: len(steps_ahead)]
                    ]
                else:
                    for ci, (name, cell) in enumerate(cells.items()):
                        if ci == 0:
                            outs = cell.propagate_steps_ahead(
                                x0_fullT[:, (self.nx * ci) : (self.nx * (ci + 1))],
                                x0_fullT[:, (self.nx * ci) : (self.nx * (ci + 1))],
                                x0_fullT[:, (self.nx * ci) : (self.nx * (ci + 1))],
                                x0T[:, (self.nx * ci) : (self.nx * (ci + 1))],
                                x0T[:, (self.nx * ci) : (self.nx * (ci + 1))],
                                x0T[:, (self.nx * ci) : (self.nx * (ci + 1))],
                                n1_in0,
                                FT0T,
                                prior_pred_step1,
                            )  # Propagate x0 through the dynamics
                        else:  # backward pass has the state that it needs to propagate from
                            outs = cell.call(
                                [
                                    x[0, 0:1, :].astype("float32")
                                    for x in predictor_input
                                ],  # Predictor input for first time step
                                [
                                    x0_fullT[
                                        :, (self.nx * ci) : (self.nx * (ci + 1))
                                    ].astype("float32")
                                ],
                            )[0]
                        outs = [np.array(o) for o in outs]
                        steps_ahead_x0s_this = [
                            np.array(x)[0:1, ...]
                            for x in outs[len(steps_ahead) : (2 * len(steps_ahead))]
                        ]
                        steps_ahead_internal_x0s_this = [
                            np.array(x)[0:1, ...]
                            for x in outs[
                                (2 * len(steps_ahead)) : (3 * len(steps_ahead))
                            ]
                        ]
                        steps_ahead_y_hat0_this = [
                            np.array(y)[0:1, ...] for y in outs[: len(steps_ahead)]
                        ]
                        if ci == 0:
                            steps_ahead_x0s = steps_ahead_x0s_this
                            steps_ahead_internal_x0s = steps_ahead_internal_x0s_this
                            steps_ahead_y_hat0 = steps_ahead_y_hat0_this
                        else:

                            def merge_directions(o1, o2, merge_mode):
                                if merge_mode == "concat":
                                    return np.concatenate([o1, o2], axis=1)
                                elif merge_mode == "sum":
                                    return o1 + o2
                                elif merge_mode == "ave":
                                    return (o1 + o2) / 2
                                elif merge_mode == "mul":
                                    return o1 * o2
                                elif merge_mode is None:
                                    return [o1, o2]
                                else:
                                    raise (Exception("Not supported"))

                            steps_ahead_x0s = [
                                merge_directions(
                                    steps_ahead_x0s[i],
                                    steps_ahead_x0s_this[i],
                                    "concat",
                                )  # Always concatenate states
                                for i in range(len(steps_ahead_x0s))
                            ]
                            steps_ahead_internal_x0s = [
                                merge_directions(
                                    steps_ahead_internal_x0s[i],
                                    steps_ahead_internal_x0s_this[i],
                                    "concat",
                                )  # Always concatenate states
                                for i in range(len(steps_ahead_internal_x0s))
                            ]
                            steps_ahead_y_hat0 = [
                                merge_directions(
                                    steps_ahead_y_hat0[i],
                                    steps_ahead_y_hat0_this[i],
                                    self.bidir_merge_mode,
                                )
                                for i in range(len(steps_ahead_x0s))
                            ]

                allXp_steps = []
                Xp_steps = []
                allYp_steps = []
                allXp_internal_steps = []
                Xp_internal_steps = []
                for saInd, step_ahead in enumerate(steps_ahead):
                    allXpThis, XpThis = prepend_x0_and_pop_last_sample(
                        x_hat_steps[saInd], steps_ahead_x0s[saInd], Ndat
                    )
                    y_hatACutThis, y_hat_next = prepend_x0_and_pop_last_sample(
                        y_hat_steps[saInd], steps_ahead_y_hat0[saInd], Ndat
                    )
                    if return_internal_states:
                        (
                            allXpInternalThis,
                            XpInternalThis,
                        ) = prepend_x0_and_pop_last_sample(
                            x_hat_internal_steps[saInd],
                            steps_ahead_internal_x0s[saInd],
                            Ndat,
                        )
                    allXp_steps.append(allXpThis.T)
                    Xp_steps.append(XpThis)
                    if return_internal_states:
                        allXp_internal_steps.append(allXpInternalThis.T)
                        Xp_internal_steps.append(XpInternalThis)

                    if not self.isOutputCategorical():
                        allYpThis = y_hatACutThis.T
                    else:
                        allYp_logits = y_hatACutThis.transpose([1, 0, 2])
                        # Convert to class probabilities by applying softmax
                        softmax = tf.keras.layers.Softmax(
                            input_shape=allYp_logits.shape
                        )
                        allYpThis = softmax(allYp_logits).numpy()

                    allYp_steps.append(allYpThis)
                outs = tuple(allXp_steps) + tuple(allYp_steps) + tuple(Xp_steps)
                if return_internal_states:
                    outs += tuple(allXp_internal_steps) + tuple(Xp_internal_steps)
            else:  # Unused (kept for reference)
                allXp, Xp = prepend_x0_and_pop_last_sample(x_hat, x0, Ndat)
                allXp = allXp.T

                # Estimate yHat_{0|-1} (first sample of output given initial state and feedthrough) and then
                # shift by one so that allYp starts from YHat_{0|-1} and ends in YHat_{Ndat|Ndat-1}
                if self.rnn.stateful:
                    self.rnn.reset_states(
                        states=initial_state
                    )  # Set the initial RNN state to zero
                if self.nft > 0 and FT_in is not None and FT_in_shift_by_one:
                    C_input_np = np.concatenate(
                        (
                            np.tile(x0[:, np.newaxis], [1, feedthrough_res.shape[0]]),
                            (
                                FT0.T
                                if FT0.shape[0] == feedthrough_res.shape[0]
                                else np.tile(FT0.T, [1, feedthrough_res.shape[0]])
                            ),
                        ),
                        axis=0,
                    ).T
                    if self.rnn.stateful:  # If not trial based
                        C_input_np = C_input_np[0:1, ...]
                else:
                    C_input_np = x0[:, np.newaxis].T
                C_input = tf.cast(C_input_np, dtype=self.rnn.cell.C.dtype)
                if self.has_prior_pred:
                    if prior_pred_shift_by_one and prior_pred is not None:
                        prior_pred_step1 = pp0_steps[0]
                    else:
                        prior_pred_step1 = None
                    C_input = (C_input, prior_pred_step1)
                y_hat0 = self.rnn.cell.C.apply_func(C_input).numpy()
                y_hatACut, y_hat_next = prepend_x0_and_pop_last_sample(
                    y_hat, y_hat0, Ndat
                )
                if not self.isOutputCategorical():
                    allYp = y_hatACut.T
                else:
                    allYp_logits = y_hatACut.transpose([1, 0, 2])
                    # Convert to class probabilities by applying softmax
                    softmax = tf.keras.layers.Softmax(input_shape=allYp_logits.shape)
                    allYp = softmax(allYp_logits).numpy()
                outs = allXp, allYp, Xp
        else:
            if self.nft > 0 or self.has_prior_pred:
                raise (
                    Exception(
                        "Not supported yet! set_batch_size to 1 to use faster prediction method instead"
                    )
                )
            allXp = np.empty((self.nx, Ndat))  # X(i|i-1)
            Xp = np.zeros((self.nx, 1))
            if self.isOutputCategorical():
                allYp = np.empty(
                    (self.ny_out, Ndat, self.cell_args["CSettings"]["num_classes"])
                )  # Y(i|i-1)
                Yp = np.zeros(
                    (self.ny_out, 1, self.cell_args["CSettings"]["num_classes"])
                )
                softmax = tf.keras.layers.Softmax(input_shape=Yp.shape)
            else:
                allYp = np.empty((self.ny_out, Ndat))  # Y(i|i-1)
                Yp = np.zeros((self.ny_out, 1))
            self.rnn.reset_states(
                states=initial_state
            )  # Set the initial RNN state to zero
            for i in range(Ndat):
                allXp[:, i] = np.squeeze(Xp)  # X(i|i-1)
                if self.isOutputCategorical():
                    allYp[:, i, :] = softmax(Yp).numpy()[:, 0, :]  # Y(i|i-1)
                else:
                    allYp[:, i] = Yp[:, 0]  # Y(i|i-1)
                Y_in_i_rep = np.tile(
                    Y_in[:, i][np.newaxis, np.newaxis, :], (self.batch_size, 1, 1)
                )
                Yp_rep, Xp_rep = self.rnn((Y_in_i_rep,))
                Xp = np.array(Xp_rep[0, :])[
                    :, np.newaxis
                ]  # First batch is the main batch
                if self.isOutputCategorical():
                    Yp = np.array(Yp_rep[0, :]).transpose(
                        [1, 0, 2]
                    )  # First batch is the main batch
                else:
                    Yp = np.array(Yp_rep[0, :]).T  # First batch is the main batch
            outs = allXp, allYp, Xp
        return outs

    def plot_comp_graph(
        self,
        savepath="model_graph",
        saveExtensions=None,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True,
    ):
        if saveExtensions is None:
            saveExtensions = ["png"]
        saveExtensions = [se for se in saveExtensions if se != "svg"]
        for fmt in saveExtensions:
            try:
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=f"{savepath}.{fmt}",
                    show_shapes=show_shapes,
                    show_layer_names=show_layer_names,
                    expand_nested=expand_nested,
                )
                logger.info(f"Saved model graph as {savepath}.{fmt}")
            except Exception as e:
                logger.error(e)
