""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Some base classes that RNNModel and RegressionModel inherit from"""

import copy
import io
import logging
import os
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .plot import plotPredictionScatter, plotTimeSeriesPrediction
from .tf_tools import (
    convertHistoryToDict,
    getModelFitHistoyStr,
    set_global_tf_eagerly_flag,
)

logger = logging.getLogger(__name__)


class ReconstructionInfo:
    """A class that can store information required to reconstruct a RegressionModel or RNNModel based on
    their constructor arguments and tf weights (rather than tf objects), which can be easily be pickled
    """

    def __init__(self, weights, constructor_kwargs):
        self.weights = weights
        self.constructor_kwargs = constructor_kwargs


class Reconstructable:
    """A class that allows a child class with tf models to be saved into pickle files and later be
    reconstructred
    """

    def get_recreation_info(self):
        constructor_kwargs = self.constructor_kwargs
        # for k in self.constructor_kwargs.keys():
        #     constructor_kwargs[k] = getattr(self, k, self.constructor_kwargs[k])
        return ReconstructionInfo(
            weights=self.model.get_weights(), constructor_kwargs=constructor_kwargs
        )

    def reconstruct(self, reconstruction_info):
        cls = type(self)  # Get the child class: either RNNModel or RegressionModel
        newInstance = cls(**reconstruction_info.constructor_kwargs)
        newInstance.model.set_weights(reconstruction_info.weights)
        return newInstance

    def save_to_file(self, file_path):
        """Calls the Keras save method to save the model to a file.

        Args:
            file_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.model.save(file_path)

    def load_from_file(self, file_path):
        """Calls the Keras load method to load the model from a file."""
        return self.model.load_model(file_path)  # TEMP, throws error!


# Inherited from https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L1744-L1891
# Changing one line to allow a min_epoch to be set so that early stopping kicks in only after that many trials
# start_from_epoch = 0 will reduce to the original functionality
# Standard in newer tf versions:
# https://github.com/keras-team/keras/commit/05d90d2a6931b5a583579cd2ef2e6932919afa63
class EarlyStoppingWithMinEpochs(tf.keras.callbacks.EarlyStopping):
    """Modified EarlyStopping class to allow a minimum number of epochs to be specified before early stopping kicks in."""

    def __init__(self, start_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch <= self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                # if self.verbose > 0:
                logger.info(
                    "Restoring model weights from the end of the best epoch: "
                    f"{self.best_epoch + 1} (stopped at {self.stopped_epoch} epochs)."
                )
                self.model.set_weights(self.best_weights)


# https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside the notebook.
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class ModelWithFitWithRetry:
    """A class that adds a fit_with_retry method to classes inheriting from it.
    Used by RNNModel and RegressionModel.
    """

    def fit_with_retry(
        self,
        init_attempts=1,
        early_stopping_patience=3,
        early_stopping_measure="loss",
        early_stopping_restore_best_weights=True,
        start_from_epoch=0,
        tb_make_prediction_plots=False,
        tb_make_prediction_scatters=False,
        tb_plot_epoch_mod=20,
        x=None,
        y=None,
        callbacks=None,
        validation_data=None,
        keep_latest_nonnan_weights=True,
        **kwargs,
    ):
        """Calls keras fit for the model, with the option to redo the fitting multiple
        times with different initializations

        Args:
            self (RegressionModel or RNNModel): the object to fit.
                    Must be fully ready to call obj.model.fit
            init_attempts (int, optional): The number of refitting attempts. Defaults to 1.
                    If more than 1, the attempt with the smallest 'loss' will be selected in the end.
            early_stopping_patience (int, optional): [description]. Defaults to 3.
            # The rest of the arguments will be passed to keras model.fit and should include:
            x: input
            y: output
            See the tf keras help for more:
            https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        Returns:
            history (A History object): the output of keras model fit
        """
        if callbacks is None:
            callbacks = []

        def compute_reg_loss(epoch, logs=None):
            if logs is None:
                return
            logs["learning_rate"] = float(
                self.model.optimizer.learning_rate
            )  # Save current learning rate
            if len(self.model.losses):  # If we have some regularization loss
                regularization_loss = float(tf.math.add_n(self.model.losses))
                logs["regularization_loss"] = regularization_loss
                total_loss = logs["loss"]
                logs["loss_minus_regularization"] = total_loss - regularization_loss
                if "val_loss" in logs:
                    logs["val_loss_minus_regularization"] = (
                        logs["val_loss"] - regularization_loss
                    )

        latest_nonnan_weights, latest_nonnan_weights_epoch, latest_epoch = (
            None,
            None,
            None,
        )

        def keep_latest_nonnan_weights_callback(epoch, logs=None):
            nonlocal latest_nonnan_weights, latest_nonnan_weights_epoch, latest_epoch
            weights = self.model.get_weights()
            nan_weights = [v for v in weights if np.any(np.isnan(v))]
            if len(nan_weights) == 0:
                latest_nonnan_weights = copy.copy(weights)
                latest_nonnan_weights_epoch = epoch
            else:
                logger.info(
                    f"epoch {epoch} has {len(nan_weights)} blown up nan weights!"
                )
            latest_epoch = epoch

        keep_latest_nonnan_weights_callback(
            0
        )  # To save initial weights, which should definitely be nonnan

        from ..DPADModel import shift_ms_to_1s_series
        from ..RegressionModel import RegressionModel

        attempt = 0
        modelsWeightsAll, historyAll, log_subdirs = [], [], []
        while attempt < init_attempts:
            attempt += 1
            if init_attempts > 1:
                logger.info(
                    "Starting fit attempt {} of {}".format(attempt, init_attempts)
                )
            callbacks_this = copy.deepcopy(callbacks)

            # Early stopping:
            early_stopping_callback = EarlyStoppingWithMinEpochs(
                monitor=early_stopping_measure,
                patience=early_stopping_patience,
                restore_best_weights=early_stopping_restore_best_weights,
                start_from_epoch=start_from_epoch,
            )
            callbacks_this.append(early_stopping_callback)
            callbacks_this.append(
                tf.keras.callbacks.LambdaCallback(on_epoch_end=compute_reg_loss)
            )

            if keep_latest_nonnan_weights:
                callbacks_this.append(
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=keep_latest_nonnan_weights_callback
                    )
                )

            if self.log_dir != "":
                log_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
                log_subdirs.append(log_subdir)
                log_dir = os.path.join(self.log_dir, log_subdir)
                logger.info("Tensorboard log_dir: {}".format(log_dir))
                callbacks_this.append(
                    tf.keras.callbacks.TensorBoard(
                        log_dir=log_dir, histogram_freq=1, profile_batch="10,20"
                    )
                )
                # # Save regularization loss
                # file_writer_metrics = tf.summary.create_file_writer(log_dir + '/metrics')
                # def tensorboard_save_reg_loss(epoch, logs):
                #     total_loss = logs['loss']
                #     if len(self.model.losses):
                #         regularization_loss = tf.math.add_n(self.model.losses)
                #     else:
                #         regularization_loss = 0
                #     with file_writer_metrics.as_default():
                #         tf.summary.scalar('regularization loss', data=regularization_loss, step=epoch)
                #         tf.summary.scalar('fit loss', data=total_loss-regularization_loss, step=epoch)
                # callbacks_this.append(tf.keras.callbacks.LambdaCallback(
                #     on_epoch_end=tensorboard_save_reg_loss
                # ))
                # Save some data plots if requested
                if tb_make_prediction_plots or tb_make_prediction_scatters:
                    file_writer_plot = tf.summary.create_file_writer(log_dir + "/plots")
                    if isinstance(self, RegressionModel):  # For RegressionModel
                        y_in = y
                        yAll = y_in
                    else:  # For RNNModel
                        y_in = y[0]
                        yAll = np.reshape(
                            y_in, (int(y_in.size / y_in.shape[-1]), y_in.shape[-1]), "C"
                        )
                    if validation_data is not None:
                        x_val, y_val = validation_data[0], validation_data[1]
                        if isinstance(self, RegressionModel):  # For RegressionModel
                            yAll_val = y_val
                        else:  # For RNNModel
                            yAll_val = np.reshape(
                                y_val[0],
                                (int(y_val[0].size / y_in.shape[-1]), y_in.shape[-1]),
                                "C",
                            )
                    ny = y_in.shape[-1]
                    nyIndsToPlot = np.array(
                        np.unique(np.round(np.linspace(0, ny - 1, 2))), "int"
                    )
                    fig1, fig2, fig3, fig4 = None, None, None, None

                    def tensorboard_plot_signals(epoch, logs):
                        nonlocal fig1, fig2, fig3, fig4
                        if (
                            epoch != 0
                            and epoch % tb_plot_epoch_mod < tb_plot_epoch_mod - 1
                        ):
                            return
                        steps_ahead = (
                            [1]
                            if not hasattr(self, "steps_ahead")
                            or self.steps_ahead is None
                            else self.steps_ahead
                        )
                        # Use the model to predict the values from the validation dataset.
                        if isinstance(self, RegressionModel):  # For RegressionModel
                            yHat = [self.model.predict(x)]
                            predLegStrs = ["Pred"]
                            batchCntStr = ""
                        else:  # For RNNModel
                            yHat = self.predict_with_keras(x)[: len(steps_ahead)]
                            yHat = [
                                np.reshape(yHatThis, yAll.shape)
                                for yHatThis in list(yHat)
                            ]
                            yHat = list(
                                shift_ms_to_1s_series(
                                    yHat,
                                    steps_ahead,
                                    missing_marker=np.nan,
                                    time_first=True,
                                )
                            )
                            predLegStrs = [
                                f"Pred {step_ahead}-step" for step_ahead in steps_ahead
                            ]
                            batch_count = int(x[0].shape[0] / self.batch_size)
                            batchCntStr = ", {} batches".format(batch_count)
                        titleHead = "Epoch:{}{} (training data)\n".format(
                            epoch, batchCntStr
                        )
                        if tb_make_prediction_plots:
                            if fig1 is not None:
                                fig1.clf()
                            plotArgs = {
                                "missing_marker": self.missing_marker,
                                "addNaNInTimeGaps": False,
                                "plotDims": nyIndsToPlot,
                                "predLegStrs": predLegStrs,
                                "y_pred_is_list": True,
                                "lineStyles": ["-", "-", "--", "-.", ":"],
                                "figsize": (11, 6),
                                "predPerfsToAdd": ["R2", "CC", "MSE"],
                                "return_fig": True,
                            }
                            fig1 = plotTimeSeriesPrediction(
                                yAll, yHat, titleHead=titleHead, fig=fig1, **plotArgs
                            )
                            plot_image = plot_to_image(fig1)
                            with file_writer_plot.as_default():
                                tf.summary.image(
                                    "Training prediction", plot_image, step=epoch
                                )
                        if tb_make_prediction_scatters:
                            if fig2 is not None:
                                fig2.clf()
                            scatterArgs = {
                                "missing_marker": self.missing_marker,
                                "plot45DegLine": True,
                                "plotLSLine": True,
                                "styles": {"size": 10, "marker": "x"},
                                "figsize": (11, 4),
                                "title": ["Dim{} ".format(di) for di in nyIndsToPlot],
                                "legNames": [
                                    f"{step_ahead}-step" for step_ahead in steps_ahead
                                ],
                                "addPerfMeasuresToLegend": ["CC", "R2"],
                                "addPerfMeasuresToTitle": ["CC", "R2", "MSE"],
                                "return_fig": True,
                            }
                            fig2 = plotPredictionScatter(
                                [yAll[..., di] for di in nyIndsToPlot],
                                [
                                    np.array([yHatStep[..., di] for yHatStep in yHat])
                                    for di in nyIndsToPlot
                                ],
                                titleHead=[titleHead] + [""] * len(nyIndsToPlot),
                                fig=fig2,
                                **scatterArgs,
                            )
                            plot_image = plot_to_image(fig2)
                            with file_writer_plot.as_default():
                                tf.summary.image(
                                    "Training prediction (scatter)",
                                    plot_image,
                                    step=epoch,
                                )
                        # The same for validation data
                        if validation_data is not None:
                            if isinstance(self, RegressionModel):  # For RegressionModel
                                yHat_val = [self.model.predict(x_val)]
                                batchCntStr = ""
                            else:  # For RNNModel
                                yHat_val = self.predict_with_keras(x_val)[
                                    : len(steps_ahead)
                                ]
                                yHat_val = [
                                    np.reshape(yHatThis, yAll_val.shape)
                                    for yHatThis in list(yHat_val)
                                ]
                                yHat_val = list(
                                    shift_ms_to_1s_series(
                                        yHat_val,
                                        steps_ahead,
                                        missing_marker=np.nan,
                                        time_first=True,
                                    )
                                )
                                batchCntStr = ", {} batches".format(
                                    int(x_val[0].shape[0] / self.batch_size)
                                )
                            titleHead = "Epoch:{}{} (validation)\n".format(
                                epoch, batchCntStr
                            )
                            if tb_make_prediction_plots:
                                if fig3 is not None:
                                    fig3.clf()
                                fig3 = plotTimeSeriesPrediction(
                                    yAll_val,
                                    yHat_val,
                                    titleHead=titleHead,
                                    fig=fig3,
                                    **plotArgs,
                                )
                                plot_image = plot_to_image(fig3)
                                with file_writer_plot.as_default():
                                    tf.summary.image(
                                        "Validation prediction", plot_image, step=epoch
                                    )
                            if tb_make_prediction_scatters:
                                if fig4 is not None:
                                    fig4.clf()
                                fig4 = plotPredictionScatter(
                                    [yAll_val[..., di] for di in nyIndsToPlot],
                                    [
                                        np.array(
                                            [yHatStep[..., di] for yHatStep in yHat_val]
                                        )
                                        for di in nyIndsToPlot
                                    ],
                                    titleHead=[titleHead] + [""] * len(nyIndsToPlot),
                                    fig=fig4,
                                    **scatterArgs,
                                )
                                plot_image = plot_to_image(fig4)
                                with file_writer_plot.as_default():
                                    tf.summary.image(
                                        "Validation prediction (scatter)",
                                        plot_image,
                                        step=epoch,
                                    )

                    callbacks_this.append(
                        tf.keras.callbacks.LambdaCallback(
                            on_epoch_end=tensorboard_plot_signals
                        )
                    )
            eagerly_flag_backup = set_global_tf_eagerly_flag(False)
            if eagerly_flag_backup:
                logger.warning(
                    "Tensorflow was set up globally to run eagerly. This is EXTREMELY slow so we have temporarily disabled it and will reenable it after model fitting. Consider fixing this global setting by running tf.config.run_functions_eagerly(False)."
                )
            if self.model.run_eagerly or tf.config.functions_run_eagerly():
                warnings.warn(
                    "This Tensorflow model is set up to run eagerly. This will be EXTREMELY slow!!! Please fix."
                )
            if len(self.model.trainable_weights) == 0:
                logger.info(f"No trainable weights... skipping training.")
            tic = time.perf_counter()
            history = self.model.fit(
                x=x,
                y=y,
                callbacks=callbacks_this,
                validation_data=validation_data,
                **kwargs,
            )
            toc = time.perf_counter()
            fitTime = toc - tic
            set_global_tf_eagerly_flag(eagerly_flag_backup)
            if hasattr(early_stopping_callback, "stopped_epoch"):
                history.params["stopped_epoch"] = early_stopping_callback.stopped_epoch
            if hasattr(early_stopping_callback, "best_epoch"):
                history.params["best_epoch"] = early_stopping_callback.best_epoch
            if early_stopping_restore_best_weights:
                picked_epoch = history.params["best_epoch"]
            else:
                picked_epoch = history.history["epoch"][-1]
            if "verbose" in kwargs and kwargs["verbose"] != 2:
                logFields = [k for k in history.history.keys()]
                logger.info(
                    "\n"
                    + getModelFitHistoyStr(history, fields=logFields, keep_ratio=0.1)
                )
                if "regularization_loss" in history.history:
                    total_loss = np.array(history.history["loss"])
                    reg_loss = np.array(history.history["regularization_loss"])
                    loss_range = np.quantile(total_loss, [0.01, 0.99])
                    reg_loss_range = np.quantile(reg_loss, [0.01, 0.99])
                    reg_to_total_change_ratio = (
                        np.diff(reg_loss_range)[0] / np.diff(loss_range)[0]
                    )
                    median_reg_to_total_ratio = np.median(reg_loss / total_loss)
                    logger.info(
                        "{:.2g}% of the changes in total loss ({:.2g} => {:.2g}) are due to changes in regularization loss ({:.2g} => {:.2g})".format(
                            reg_to_total_change_ratio * 100,
                            loss_range[1],
                            loss_range[0],
                            reg_loss_range[1],
                            reg_loss_range[0],
                        )
                    )
                    logger.info(
                        "Median ratio of reg_loss to total_loss is {:.2g}%".format(
                            median_reg_to_total_ratio * 100
                        )
                    )
                    if np.any((total_loss - reg_loss) < 0):
                        logger.info("Loss has negative values")
                        reg_to_loss_ratio = reg_to_total_change_ratio
                    else:
                        reg_to_loss_ratio = median_reg_to_total_ratio
                    if reg_to_loss_ratio > 0.5:
                        logger.info(
                            "Regularization lambda is too high, regularization is dominating the total loss"
                        )
                    elif reg_to_loss_ratio < 0.01:
                        logger.info(
                            "Regularization lambda is too low, regularization is an almost negligible part (<1%) of the total loss"
                        )
            logger.info("Model fitting took {:.2f}s".format(fitTime))
            weights = self.model.get_weights()
            nan_weights = [v for v in weights if np.any(np.isnan(v))]
            if (
                len(nan_weights) > 0
                and keep_latest_nonnan_weights
                and latest_nonnan_weights is not None
            ):
                logger.warning(
                    f"{len(nan_weights)} weights had nans, replacing with weights from the latest epoch with non-nan weights (epoch {latest_nonnan_weights_epoch})"
                )
                self.model.set_weights(latest_nonnan_weights)
                epoch_ind = [
                    ep for ep in history.epoch if ep <= latest_nonnan_weights_epoch
                ][-1]
                for key in history.history:
                    history.history[key][-1] = history.history[key][epoch_ind]
                picked_epoch = history.epoch[epoch_ind]
            history.params["picked_epoch"] = picked_epoch
            if init_attempts > 1:
                weights = self.model.get_weights()
                modelsWeightsAll.append(weights)
                historyAll.append(history)
                self.build()
                # Reset model weights
                if attempt == init_attempts:  # Select final model
                    lossAll = [
                        np.array(h.history[early_stopping_measure])[
                            np.where(np.array(h.epoch) == h.params["picked_epoch"])[0]
                        ][0]
                        for h in historyAll
                    ]
                    if np.all(np.isnan(lossAll)):
                        msg = "All fit attempts ended up with a nan loss (probably blew up)!"
                        if not keep_latest_nonnan_weights:
                            msg += " Consider setting keep_latest_nonnan_weights=True to keep latest epoch with non-nan loss in case of blow up."
                        # raise(Exception(msg))
                        logger.warning(msg)
                    if np.all(np.isnan(lossAll)):
                        logger.warning(
                            "All attempts resulted in NaN loss for all epochs!! Keeping initial random params from attempt 1. "
                        )
                        bestInd = 0
                    else:
                        bestInd = np.nanargmin(lossAll)
                    logger.info(
                        "Selected model from learning attempt {}/{}, which had the smallest loss ({:.8g})".format(
                            1 + bestInd, init_attempts, lossAll[bestInd]
                        )
                    )
                    self.model.set_weights(modelsWeightsAll[bestInd])
                    history = historyAll[bestInd]
                    history.params["history_all"] = [
                        convertHistoryToDict(h) for h in historyAll
                    ]
                    history.params["selected_ind"] = bestInd
                    if self.log_dir != "":
                        self.log_subdir = log_subdirs[bestInd]
            if self.log_dir != "" and (
                tb_make_prediction_plots or tb_make_prediction_scatters
            ):
                plt.close("all")
                del fig1, fig2, fig3, fig4
        return history
