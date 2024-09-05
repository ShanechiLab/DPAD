""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""
An object for building and applying linear mappings
"""

import logging

import numpy as np

from .tools import applyGivenScaling, isFlat, learnScaling, undoScaling

logger = logging.getLogger(__name__)


class LinearMapping:
    """Implements a mapping in the form of f(x) = W x + b"""

    def __init__(self, W=None, b=None, missing_marker=None):
        self.set_params(W, b)
        self.removed_inds = None
        self.to_replace_vals = []
        self.replacement_vals = []
        self.missing_marker = None

    def set_params(self, W=None, b=None):
        self.set_weight(W)
        self.set_intercept(b)

    def set_intercept(self, b=None):
        self.b = b

    def set_weight(self, W=None):
        self.W = W
        if W is not None:
            self._W_pinv = np.linalg.pinv(self.W)
        else:
            self._W_pinv = None

    def get_overall_W(self):
        return self.W

    def set_to_dimension_remover(self, keep_vector):
        eye = np.eye(keep_vector.size)
        remove_inds = np.where(~np.array(keep_vector, dtype=bool))[0]
        self.set_weight(np.delete(eye, remove_inds, 0))
        self.set_intercept(None)
        self.removed_inds = remove_inds

    def set_value_replacements(self, to_replace_vals, replacement_vals):
        self.to_replace_vals = to_replace_vals
        self.replacement_vals = replacement_vals

    def apply(self, x):
        """Applies mapping to a series of samples. Second dimension is the dimension of samples.

        Args:
            x (np.array): _description_

        Returns:
            _type_: _description_
        """
        out = x
        if hasattr(self, "to_replace_vals"):
            for ri, rep_val in enumerate(self.to_replace_vals):
                if np.isnan(rep_val):
                    rep_inds = np.isnan(out)
                elif np.isinf(rep_val):
                    rep_inds = np.isinf(out)
                else:
                    rep_inds = out == rep_val
                rep_val = (
                    self.replacement_vals[ri % len(self.replacement_vals)]
                    if isinstance(self.replacement_vals, (list, tuple))
                    else self.replacement_vals
                )
                out[rep_inds] = rep_val
        if self.W is not None:
            out = self.W @ out
        if self.b is not None:
            out = out + self.b
        return out

    def apply_inverse(self, x):
        out = x
        if self.b is not None:
            out = out - self.b
        if self.W is not None:
            out = self._W_pinv @ out
        return out


class LinearMappingPerDim(LinearMapping):
    def __init__(self, axis=0, **kw_args):
        super().__init__(**kw_args)
        self.axis = axis

    def set_weight(self, W=None):
        self.W = W

    def get_overall_W(self):
        if isinstance(self.W, (np.ndarray)):
            WMat = self.W
        else:
            WMat = np.array([self.W])
        if WMat.size < self.b.size:
            WMat = WMat * np.ones_like(self.b)
        if len(WMat.shape) == 1:
            WMat = np.diag(WMat)
        return WMat

    def set_to_zscorer(
        self,
        Y,
        axis=0,
        remove_mean=True,
        zscore=True,
        zscore_per_dim=True,
        missing_marker=None,
    ):
        if missing_marker is not None:
            self.missing_marker = missing_marker
        yMean, yStd = learnScaling(
            Y,
            remove_mean,
            zscore,
            zscore_per_dim=zscore_per_dim,
            missing_marker=self.missing_marker,
            axis=axis,
        )
        if not zscore_per_dim:
            yStd = yStd[0]
        self.axis = axis
        self.b = yMean
        self.W = yStd

    def apply(self, x):
        """Applies mapping to a series of samples. Second dimension is the dimension of samples.

        Args:
            x (np.array): _description_

        Returns:
            _type_: _description_
        """
        return applyGivenScaling(
            x, self.b, self.W, axis=self.axis, missing_marker=self.missing_marker
        )

    def apply_inverse(self, x):
        return undoScaling(
            self,
            x,
            meanField="b",
            stdField="W",
            axis=self.axis,
            missing_marker=self.missing_marker,
        )


class LinearMappingSequence:
    def __init__(self):
        self.maps = []

    def append(self, map):
        if map is not None:
            self.maps.append(map)

    def get_overall_W(self):
        if len(self.maps) == 0:
            return None
        for mi, map in enumerate(self.maps):
            thisW = map.get_overall_W()
            if mi == 0:
                W = thisW
            else:
                W = thisW @ W
        return W

    def apply(self, Y):
        out = Y
        for map in self.maps:
            out = map.apply(out)
        return out

    def apply_inverse(self, Y):
        out = Y
        for map in reversed(self.maps):
            out = map.apply_inverse(out)
        return out


def getNaNRemoverMapping(Y, signal_name="", axis=0, verbose=False):
    """Returns a LinearMapping that removes NaN/Inf dimensions of the given data data

    Args:
        Y (np.array): input data
        signal_name (str, optional): _description_. Defaults to ''.
        axis (int, optional): Axis over which to check flatness. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # Detect and remove flat data dimensions
    if Y is not None:
        isAllNans = np.all(np.isnan(Y), axis=axis)
        isAllInfs = np.all(np.isinf(Y), axis=axis)
        isBadY = np.logical_or(isAllNans, isAllInfs)
        if np.any(isBadY):
            if verbose:
                logger.warning(
                    "Warning: {}/{} dimensions of signal {} (dims: {})) were just NaN/Inf values, removing them as a preprocessing".format(
                        np.sum(isBadY), isBadY.size, signal_name, np.where(isBadY)[0]
                    )
                )
            YPrepMap = LinearMapping()
            YPrepMap.set_to_dimension_remover(~isBadY)
        else:
            YPrepMap = None
    else:
        YPrepMap = None
    return YPrepMap


def getFlatRemoverMapping(Y, signal_name="", axis=0, verbose=False):
    """Returns a LinearMapping that removes flat dimensions of the given data data

    Args:
        Y (np.array): input data
        signal_name (str, optional): _description_. Defaults to ''.
        axis (int, optional): Axis over which to check flatness. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # Detect and remove flat data dimensions
    if Y is not None:
        isFlatY = isFlat(Y, axis=axis)
        isAllNans = np.all(np.isnan(Y), axis=axis)
        isAllInfs = np.all(np.isinf(Y), axis=axis)
        isAllNaNsOrInfs = np.logical_or(isAllNans, isAllInfs)
        isFlatY = np.logical_or(isFlatY, isAllNaNsOrInfs)
        if np.any(isFlatY):
            if verbose:
                logger.warning(
                    "Warning: {}/{} dimensions of signal {} (dims: {})) were flat, removing them as a preprocessing".format(
                        np.sum(isFlatY), isFlatY.size, signal_name, np.where(isFlatY)[0]
                    )
                )
            YPrepMap = LinearMapping()
            YPrepMap.set_to_dimension_remover(~isFlatY)
        else:
            YPrepMap = None
    else:
        YPrepMap = None
    return YPrepMap


def getZScoreMapping(
    Y,
    signal_name="",
    axis=0,
    verbose=False,
    remove_mean=True,
    zscore=True,
    zscore_per_dim=True,
    missing_marker=None,
):
    """Returns a LinearMapping that zscores the given data data

    Args:
        Y (np.array): input data
        signal_name (str, optional): _description_. Defaults to ''.
        axis (int, optional): Axis over which to check flatness. Defaults to 0.

    Returns:
        _type_: _description_
    """

    if Y is not None:
        YPrepMap = LinearMappingPerDim()
        YPrepMap.set_to_zscorer(
            Y,
            axis=axis,
            remove_mean=remove_mean,
            zscore=zscore,
            zscore_per_dim=zscore_per_dim,
            missing_marker=missing_marker,
        )
    else:
        YPrepMap = None
    return YPrepMap
