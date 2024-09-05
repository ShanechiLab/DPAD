""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""RegressionModel class, which implements a flexible multi-layer perceptron (MLP)"""
"""For mathematical description see RegressionModelDoc.md"""

import copy
import io
import logging
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .tools.abstract_classes import PredictorModel
from .tools.model_base_classes import ModelWithFitWithRetry, Reconstructable
from .tools.tf_losses import (
    masked_CategoricalCrossentropy,
    masked_CC,
    masked_mse,
    masked_PoissonLL_loss,
    masked_R2,
)
from .tools.tf_tools import set_global_tf_eagerly_flag
from .tools.tools import autoDetectSignalType, get_one_hot, getIsOk

logger = logging.getLogger(__name__)


class RegressionModel(tf.keras.layers.Layer, ModelWithFitWithRetry, Reconstructable):
    """A multilayer perceptron model made from a series of fully connected layers."""

    def __init__(
        self,
        n_in,
        n_out,
        units=[],
        use_bias=False,
        dropout_rate=0,  # If non-zero, hidden layers will have this much dropout (can also be a list for each layer)
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer_name=None,  # Name of tf.keras.regularizers to use (options: 'l1', 'l2', 'l1_l2')
        kernel_regularizer_args={},  # Dict of arguments to pass to the regularizer
        bias_regularizer_name=None,  # Name of tf.keras.regularizers to use
        bias_regularizer_args={},  # Dict of arguments to pass to the regularizer
        activation="linear",
        output_activation=None,  # For the regression case, default: 'linear', except for out_dist='poisson' that will have 'exponential'
        num_classes=None,  # If provided, will do classification
        out_dist=None,  # Output distribution, default: None which means 'gaussian', can also be 'poisson'
        has_prior_pred=False,  # If true, will support a prior prediction during training
        prior_pred_op=None,  # Operation required to combine prior pred. None gives the default which is
        # 'add' for gaussian and categorical dists, and 'multiply' for out_dist='poisson'
        name="reg_",
        log_dir="",  # If not empty, will store tensorboard logs
        optimizer_name="Adam",  # Name of optimizer
        optimizer_args=None,  # Dict of arguments for the optimizer
        lr_scheduler_name=None,  # The name of learning rate scheduler to use, e.g., ExponentialDecay
        lr_scheduler_args=None,  # The arguments of the learning rate scheduler to use
        missing_marker=None,
    ):
        self.constructor_kwargs = {
            "n_in": n_in,
            "n_out": n_out,
            "units": units,
            "dropout_rate": dropout_rate,
            "use_bias": use_bias,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "kernel_regularizer_name": kernel_regularizer_name,
            "kernel_regularizer_args": kernel_regularizer_args,
            "bias_regularizer_name": bias_regularizer_name,
            "bias_regularizer_args": bias_regularizer_args,
            "activation": activation,
            "output_activation": output_activation,
            "num_classes": num_classes,
            "out_dist": out_dist,
            "has_prior_pred": has_prior_pred,
            "prior_pred_op": prior_pred_op,
            "name": name,
            "log_dir": log_dir,
            "optimizer_name": optimizer_name,
            "optimizer_args": optimizer_args,
            "lr_scheduler_name": lr_scheduler_name,
            "lr_scheduler_args": lr_scheduler_args,
            "missing_marker": missing_marker,
        }
        super(RegressionModel, self).__init__(name=name)

        def ensure_is_list(v):
            v = copy.deepcopy(v)
            if "ListWrapper" in str(type(v)):
                v = list(v)
            if type(v) is tuple:
                v = list(v)
            if type(v) not in [list, tuple]:
                v = [v]
            return v

        units = ensure_is_list(units)

        units.append(n_out)  # Add output later

        if output_activation is None:
            output_activation = "linear" if out_dist != "poisson" else "exponential"

        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = len(units)
        self.layers = []
        self.units = units
        self.dropout_rate = ensure_is_list(dropout_rate)
        self.use_bias = ensure_is_list(use_bias)
        self.kernel_initializer = ensure_is_list(kernel_initializer)
        self.bias_initializer = ensure_is_list(bias_initializer)
        self.kernel_regularizer_name = ensure_is_list(kernel_regularizer_name)
        self.kernel_regularizer_args = ensure_is_list(kernel_regularizer_args)
        self.bias_regularizer_name = ensure_is_list(bias_regularizer_name)
        self.bias_regularizer_args = ensure_is_list(bias_regularizer_args)
        self.activation = ensure_is_list(activation)
        self.output_activation = output_activation
        self.num_classes = num_classes
        self.out_dist = out_dist
        self.has_prior_pred = has_prior_pred
        if prior_pred_op is None:
            prior_pred_op = "multiply" if self.out_dist == "poisson" else "add"
        self.prior_pred_op = prior_pred_op
        self.name_prefix = name
        self.log_dir = log_dir
        self.logsub_dir = ""
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_args = (
            lr_scheduler_args if lr_scheduler_args is not None else {}
        )
        self.missing_marker = missing_marker
        self.build()

    def build(self):
        def get_nth_or_last_elem(L, ci):
            return L[int(np.min((ci, len(L) - 1)))]

        self.inputs = tf.keras.Input(
            shape=(self.n_in,), name="{}input".format(self.name_prefix)
        )
        x = self.inputs
        for ci in range(self.num_layers):
            if ci == (self.num_layers - 1):
                thisActivation = self.output_activation
            elif len(self.activation) > ci:
                thisActivation = self.activation[ci]
            else:
                thisActivation = self.activation[-1]
            kernel_regularizer = get_nth_or_last_elem(self.kernel_regularizer_name, ci)
            if kernel_regularizer is not None:
                args = get_nth_or_last_elem(self.kernel_regularizer_args, ci)
                kernel_regularizer = getattr(tf.keras.regularizers, kernel_regularizer)(
                    **args
                )
            bias_regularizer = get_nth_or_last_elem(self.bias_regularizer_name, ci)
            if bias_regularizer is not None:
                args = get_nth_or_last_elem(self.bias_regularizer_args, ci)
                bias_regularizer = getattr(tf.keras.regularizers, bias_regularizer)(
                    **args
                )
            nUnits = copy.copy(self.units[ci])
            if self.num_classes is not None and ci == (
                self.num_layers - 1
            ):  # Output layer for classification
                nUnits = nUnits * self.num_classes
            thisLayer = tf.keras.layers.Dense(
                nUnits,
                use_bias=get_nth_or_last_elem(self.use_bias, ci),
                kernel_initializer=get_nth_or_last_elem(self.kernel_initializer, ci),
                bias_initializer=get_nth_or_last_elem(self.bias_initializer, ci),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation=thisActivation,
                name="{}dense_{}".format(self.name_prefix, ci + 1),
            )
            self.layers.append(thisLayer)
            x = thisLayer(x)
            this_dropout_rate = get_nth_or_last_elem(self.dropout_rate, ci)
            if this_dropout_rate > 0 and (
                ci < (self.num_layers - 1) or len(self.dropout_rate) >= self.num_layers
            ):
                dropout = tf.keras.layers.Dropout(
                    this_dropout_rate,
                    name="{}dropout_{}".format(self.name_prefix, ci + 1),
                )
                self.layers.append(dropout)
                x = dropout(x)
            if self.num_classes is not None and ci == (
                self.num_layers - 1
            ):  # Output layer for classification
                x = tf.reshape(x, [-1, self.units[ci], self.num_classes])
        self.outputs = x
        if self.has_prior_pred:
            if self.num_classes is not None:
                self.prior_pred = tf.keras.Input(
                    shape=(
                        self.n_out,
                        self.num_classes,
                    ),
                    name="{}prior_pred".format(self.name_prefix),
                )
            else:
                self.prior_pred = tf.keras.Input(
                    shape=(self.n_out,), name="{}prior_pred".format(self.name_prefix)
                )
            if self.prior_pred_op == "add":
                self.outputs = (
                    self.outputs + self.prior_pred
                )  # Here we have either logits or Gaussian vars so they add up in both cases
            elif self.prior_pred_op == "multiply":
                self.outputs = (
                    self.outputs * self.prior_pred
                )  # Useful for outdist=poisson with exponential activation function
            else:
                raise (Exception("Not supported"))
            self.model = tf.keras.models.Model(
                inputs=[self.inputs, self.prior_pred], outputs=self.outputs
            )
        else:
            self.model = tf.keras.models.Model(inputs=self.inputs, outputs=self.outputs)
        self.compile()

    def compile(self):
        metrics = []
        if self.num_classes is not None:
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
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.run_eagerly = False  # You can temporarily set this to True to enable eager execution so that you can put breakpoints inside the model for debugging. But eager execution will be EXTREMELY slow.

    def get_config(self):
        config = super(RegressionModel, self).get_config()
        initArgNames = [
            "n_in",
            "n_out",
            "units",
            "kernel_initializer",
            "kernel_regularizer_name",
            "kernel_regularizer_args" "use_bias",
            "bias_regularizer_name",
            "bias_regularizer_args",
            "activation",
            "output_activation",
            "num_classes",
            "out_dist",
            "has_prior_pred",
            "prior_pred_op",
            "log_dir",
            "missing_marker",
        ]
        for fName in initArgNames:
            config[fName] = getattr(self, fName)
        config.update({"name": self.name_prefix})
        return config

    def apply_func(self, inputs, name_scope=None):
        with tf.name_scope(
            self.name_prefix if name_scope is None else name_scope
        ) as scope:
            if self.has_prior_pred:
                out = inputs[0]
                prior_pred = inputs[1]
            else:
                out = inputs
            for ci, layer in enumerate(self.layers):
                out = layer(out)
                if self.num_classes is not None and ci == (
                    self.num_layers - 1
                ):  # Output layer for classification
                    out = tf.reshape(out, [-1, self.units[ci], self.num_classes])
            if self.has_prior_pred and prior_pred is not None:
                if self.prior_pred_op == "add":
                    check_op = tf.Assert(
                        ~tf.math.reduce_all(prior_pred == 1),
                        [
                            "Prior is always exactly 1! Might be a bug in passing the prior!"
                        ],
                        name="CHECK_NONONE",
                    )
                    out += prior_pred  # Here we have either logits or Gaussian vars so they add up in both cases
                elif self.prior_pred_op == "multiply":
                    check_op = tf.Assert(
                        ~tf.math.reduce_any(prior_pred < 0),
                        [
                            "Prior is negative for poisson! Probably a bug in passing the prior!",
                            prior_pred,
                        ],
                        name="CHECK_NONNEGATIVE",
                    )
                    check_op2 = tf.Assert(
                        ~tf.math.reduce_all(prior_pred == 0),
                        [
                            "Prior is always zero for poisson! Probably a bug in passing the prior!"
                        ],
                        name="CHECK_NONZERO",
                    )
                    out *= prior_pred  # Useful for outdist=poisson with exponential activation function
                else:
                    raise (Exception("Not supported"))
        return out

    def setTrainable(self, trainable):
        self.trainable = trainable
        self.compile()  # "make sure to call compile() again on your model for your changes to be taken into account."

    def fit(
        self,
        X_in,
        X_out,
        prior_pred=None,
        X_in_val=None,
        X_out_val=None,
        prior_pred_val=None,
        epochs=100,
        batch_size=None,
        verbose=False,
        init_attempts=1,  # Number of initialization retries for each model fitting attempt. Will keep the best outcome after each series of attempts
        max_attempts=1,  # Maximum number of times that the whole model fitting will be repeated in case of a blow-up or nan loss
        early_stopping_patience=3,
        start_from_epoch=0,
        early_stopping_measure="loss",
    ):
        """
        Inputs:
        - (1) X_in: input data. Expected dimensions: dim x samples
        - (2) X_out: output data. Expected dimensions: dim x samples
        """

        def prep_IO_data(X_in, X_out, prior_pred, goal_label_for_log="training"):
            if self.missing_marker is not None:
                isNotMissing = np.logical_not(
                    np.any(X_out == self.missing_marker, axis=0)
                )
                if not np.all(isNotMissing):
                    logger.info(
                        "Only {}/{} samples ({:.3g}%) can be used for {} (rest is missing, i.e. marker:{})".format(
                            np.sum(isNotMissing),
                            len(isNotMissing),
                            100 * np.sum(isNotMissing) / len(isNotMissing),
                            goal_label_for_log,
                            self.missing_marker,
                        )
                    )
            else:
                isNotMissing = np.ones(X_out.shape[1], dtype=bool)

            outputs = X_out[:, isNotMissing].T
            if self.num_classes is not None:
                outputs = get_one_hot(np.array(outputs, dtype=int), self.num_classes)
            inputs = X_in[:, isNotMissing].T
            if self.has_prior_pred:
                if prior_pred is None:
                    if self.num_classes is not None:  # Default is equiprobable
                        prior_pred = (
                            np.ones((X_out.shape[0], X_out.shape[1], self.num_classes))
                            / self.num_classes
                        )
                    else:  # Default is zero
                        prior_pred = np.zeros_like(X_out)
                if self.num_classes is not None:
                    prior_pred_sel = prior_pred[:, isNotMissing, :].transpose([1, 0, 2])
                    prior_pred_sel = np.log(prior_pred_sel)  # To invert the softmax
                else:
                    prior_pred_sel = prior_pred[:, isNotMissing].T
                inputs = [inputs, prior_pred_sel]

            return inputs, outputs

        inputs, outputs = prep_IO_data(X_in, X_out, prior_pred)

        if X_in_val is None:
            validation_data = None
        else:
            inputs_val, outputs_val = prep_IO_data(
                X_in_val, X_out_val, prior_pred_val, goal_label_for_log="validation"
            )
            validation_data = (inputs_val, outputs_val)

        fitWasOk, attempt = False, 0
        while not fitWasOk and attempt < max_attempts:
            attempt += 1
            history = self.fit_with_retry(
                init_attempts=init_attempts,
                early_stopping_patience=early_stopping_patience,
                early_stopping_measure=early_stopping_measure,
                start_from_epoch=start_from_epoch,
                # The rest of the arguments will be passed to keras model.fit
                x=inputs,
                y=outputs,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=verbose,
            )
            fitWasOk = not np.isnan(history.history["loss"][-1])
            if not fitWasOk and attempt < max_attempts:
                logger.info(
                    f"Regression model fit led to nan loss. Retrying with attempt {attempt+1}"
                )
        num_batch = history.params["steps"]
        logger.info(
            "Model fitting finished. Had {} batches each with batch_size of {} samples each (ny_in={}, ny_out={})".format(
                num_batch,
                int(outputs.shape[0] / num_batch),
                X_in.shape[0],
                X_out.shape[0],
            )
        )
        return history

    def predict(self, X_in, prior_pred=None):
        eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        if self.has_prior_pred and prior_pred is None:
            if self.num_classes is not None:  # Default is equiprobable
                prior_pred = (
                    np.ones((self.n_out, X_in.shape[1], self.num_classes))
                    / self.num_classes
                )
            else:  # Default is zero
                prior_pred = np.zeros((self.n_out, X_in.shape[1]))
        if not hasattr(
            self.model, "_predict_counter"
        ):  # Temp to fix new Keras error when using old models to predict:
            from tensorflow.keras.backend import variable

            self.model._predict_counter = variable(0)
        if self.num_classes is not None:
            if self.has_prior_pred:
                prior_pred = np.log(prior_pred)  # To invert the softmax
                X_out = self.model.predict([X_in.T, prior_pred.transpose([1, 0, 2])])
            else:
                X_out = self.model.predict(X_in.T)
            X_out_softmax = tf.keras.layers.Softmax(input_shape=X_out.shape)(
                X_out
            ).numpy()
            X_out = np.transpose(X_out_softmax, [1, 0, 2])
        else:
            if self.has_prior_pred:
                X_out = self.model.predict([X_in.T, prior_pred.T]).T
            else:
                X_out = self.model.predict(X_in.T).T
        set_global_tf_eagerly_flag(eagerly_flag_backup)
        return X_out

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


class DRModel(PredictorModel):
    """A class that implements non-linear direction regression model based on RegressionModel"""

    def __init__(self, log_dir=""):  # If not empty, will store tensorboard logs
        self.log_dir = log_dir

    @staticmethod
    def parse_method_code(
        methodCode, YType=None, ZType=None, Z=None, missing_marker=None
    ):
        Dz_args = {}
        Dy_args = {}
        if "HL" in methodCode:
            regex = r"([Dz|Dy|]*)(\d+)HL(\d+)U"  # 1HL100U
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                var_names, hidden_layers, hidden_units = match.groups()
            hidden_layers = int(hidden_layers)
            hidden_units = int(hidden_units)
            activation = "relu"
            NL_args = {
                "use_bias": True,
                "units": [hidden_units] * hidden_layers,
                "activation": activation,
            }
            if var_names == "" or "Dz" in var_names:
                Dz_args = copy.deepcopy(NL_args)
            if var_names == "" or "Dy" in var_names:
                Dy_args = copy.deepcopy(NL_args)
        if "RGL" in methodCode:  # Regularize
            regex = r"([Dz|Dy|]*)RGL(\d+)"  #
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                var_names, norm_num = match.groups()
            if norm_num in ["1", "2"]:
                regularizer_name = "l{}".format(norm_num)
            else:
                raise (Exception("Unsupported method code: {}".format(methodCode)))
            lambdaVal = 0.01  # Default: 'l': 0.01
            regex = r"L(\d+)e([-+])?(\d+)"  # 1e-2
            matches = re.finditer(regex, methodCode)
            for matchNum, match in enumerate(matches, start=1):
                m, sgn, power = match.groups()
                if sgn is not None and sgn == "-":
                    power = -float(power)
                lambdaVal = float(m) * 10 ** float(power)
            regularizer_args = {"l": lambdaVal}  # Default: 'l': 0.01
            RGL_args = {
                "kernel_regularizer_name": regularizer_name,
                "kernel_regularizer_args": regularizer_args,
                "bias_regularizer_name": regularizer_name,
                "bias_regularizer_args": regularizer_args,
            }
            if var_names == "" or "Dz" in var_names:
                Dz_args.update(copy.deepcopy(RGL_args))
            if var_names == "" or "Dy" in var_names:
                Dy_args.update(copy.deepcopy(RGL_args))

        if ZType == "count_process":
            Dz_args["use_bias"] = True
            Dz_args["out_dist"] = "poisson"
            Dz_args["output_activation"] = "exponential"
        elif ZType == "cat":
            isOkZ = getIsOk(Z, missing_marker)
            ZClasses = np.unique(Z[:, np.all(isOkZ, axis=0)])
            Dz_args["num_classes"] = len(ZClasses)
            Dz_args["use_bias"] = True

        return Dy_args, Dz_args

    def fit(
        self,
        Y,
        Z=None,
        U=None,
        batch_size=32,  # Each batch consists of this many blocks with block_samples time steps
        epochs=250,  # Max number of epochs to go over the whole training data
        Y_validation=None,  # if provided will use to compute loss on validation
        Z_validation=None,  # if provided will use to compute loss on validation
        U_validation=None,  # if provided will use to compute loss on validation
        true_model=None,
        missing_marker=None,  # Values of z that are equal to this will not be used
        allowNonzeroCz2=True,
        model2_Cz_Full=True,
        skip_Cy=False,  # If true and only stage 1 (n1 >= nx), will not learn Cy (model will not have neural self-prediction ability)
        clear_graph=True,  # If true will wipe the tf session before starting, so that variables names don't get numbers at the end and mem is preserved
        YType=None,
        ZType=None,
        Dy_args={},
        Dz_args={},
    ):
        if clear_graph:
            tf.keras.backend.clear_session()

        isOkY = getIsOk(Y, missing_marker)
        isOkZ = getIsOk(Z, missing_marker)
        isOkU = getIsOk(U, missing_marker)

        if YType is None:  # Auto detect signal types
            YType = autoDetectSignalType(Y)

        if ZType is None:  # Auto detect signal types
            ZType = autoDetectSignalType(Z)

        if ZType == "cat":
            ZClasses = np.unique(Z[:, np.all(isOkZ, axis=0)])

        if YType == "count_process":
            yDist = "poisson"
        else:
            yDist = None

        if U is None:
            U = np.empty(0)
        nu = U.shape[0]

        ny, Ndat = Y.shape[0], Y.shape[1]
        if Z is not None:
            nz, NdatZ = Z.shape[0], Z.shape[1]

        if nu > 0:
            YU = np.concatenate([Y, U], axis=0)
            if Y_validation is not None:
                YU_validation = np.concatenate([Y_validation, U_validation], axis=0)
            else:
                YU_validation = None
        else:
            YU = Y
            YU_validation = Y_validation

        logger.info("Learning regression")
        this_log_dir = "" if self.log_dir == "" else os.path.join(self.log_dir, "Dz")
        reg_args = copy.deepcopy(Dz_args)
        model_Dz = RegressionModel(
            ny + nu, nz, log_dir=this_log_dir, missing_marker=missing_marker, **reg_args
        )
        history_Dz = model_Dz.fit(
            YU, Z, X_in_val=YU_validation, X_out_val=Z_validation, epochs=epochs
        )
        self.logs = {"model_Dz": history_Dz}

        self.Dz_args = Dz_args
        self.Dy_args = Dy_args

        self.ny = ny
        self.nz = nz
        self.nu = nu

        self.model_Dz = model_Dz

        self.missing_marker = missing_marker
        self.batch_size = batch_size

    def discardModels(self):
        if self.nz > 0:
            self.model_Dz = self.model_Dz.model.get_weights()

    def restoreModels(self):
        if self.nz > 0:
            w = self.model_Dz
            reg_args = copy.deepcopy(self.Dz_args)
            self.model_Dz = RegressionModel(
                self.ny + self.nu,
                self.nz,
                missing_marker=self.missing_marker,
                **reg_args,
            )
            self.model_Dz.model.set_weights(w)

    def predict(self, Y, U=None):
        """
        Y: sample x ny
        U: sample x nu
        """

        eagerly_flag_backup = set_global_tf_eagerly_flag(False)
        Ndat = Y.shape[0]

        if U is None and self.nu > 0:
            U = np.zeros((Ndat, self.nu))
        if self.nu > 0:
            YU = np.concatenate([Y, U], axis=1)
        else:
            YU = Y

        allZp = self.model_Dz.predict(YU.T)
        allYp = None
        allXp = None

        if self.nz > 0 and allZp is not None:
            if len(allZp.shape) == 2:
                allZp = allZp.T
            else:
                allZp = allZp.transpose([1, 0, 2])

        set_global_tf_eagerly_flag(eagerly_flag_backup)
        return allZp, allYp, allXp
