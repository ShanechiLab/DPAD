""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

""" Tools for evaluating system identification """

import copy
import logging
import warnings

import numpy as np
import sklearn.metrics as metrics
from scipy.stats import wilcoxon

from .tools import applyScaling, undoScaling

logger = logging.getLogger(__name__)


def runPredict(
    sId,
    Y=None,
    Z=None,
    U=None,
    YType=None,
    ZType=None,
    useXFilt=False,
    missing_marker=None,
    undo_scaling=False,
):
    """Runs the model prediction after applying appropriate preprocessing (e.g. zscoring) on the input data and also
    undoes the preprocessing (e.g. zscoring) in the predicted data.

    Args:
        sId (PredictorModel): a model that implements a predict method.
        Y (np.array): input data. Defaults to None.
        Z (np.array): output data. Defaults to None.
        U (np.array, optional): external input data. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        useXFilt (bool, optional): if true, will pass to predict if the
            model supports that argument (i.e. is an LSSM). Defaults to False.
        missing_marker (numpy value, optional): indicator of missing samples in data.
            Is used in performing and undoing preprocessing. Defaults to None.
        undo_scaling (bool, optional): if true, will apply the inverse scaling
            on predictions. Defaults to False.

    Returns:
        zPred (np.array): predicted Z
        yPred (np.array): predicted Y
        xPred (np.array): latent state X
        Y (np.array): updated Y after any preprocessing/undoing
        Z (np.array): updated Z after any preprocessing/undoing
        U (np.array): updated U after any preprocessing/undoing
    """
    # ‌Apply any necessary scaling
    if YType == "cont":
        Y = applyScaling(sId, Y, "yMean", "yStd", missing_marker=missing_marker)
    if ZType == "cont" and Z is not None:
        Z = applyScaling(sId, Z, "zMean", "zStd", missing_marker=missing_marker)
    if U is not None:
        U = applyScaling(sId, U, "uMean", "uStd", missing_marker=missing_marker)

    # Evaluate decoding on test data
    additionalArgs = {}
    if "PSID.LSSM.LSSM" in str(type(sId)):
        additionalArgs["useXFilt"] = useXFilt
    if isinstance(Y, (list, tuple)):
        zPred, yPred, xPred = [], [], []
        for trialInd in range(len(Y)):
            zPredThis, yPredThis, xPredThis = sId.predict(
                Y[trialInd], U[trialInd] if U is not None else None, **additionalArgs
            )
            zPred.append(zPredThis)
            yPred.append(yPredThis)
            xPred.append(xPredThis)
    else:
        zPred, yPred, xPred = sId.predict(Y, U, **additionalArgs)

    if undo_scaling:
        if YType == "cont":
            yPred = undoScaling(
                sId, yPred, "yMean", "yStd", missing_marker=missing_marker
            )
            Y = undoScaling(sId, Y, "yMean", "yStd", missing_marker=missing_marker)
        if ZType == "cont":
            zPred = undoScaling(
                sId, zPred, "zMean", "zStd", missing_marker=missing_marker
            )
            Z = undoScaling(sId, Z, "zMean", "zStd", missing_marker=missing_marker)
        if U is not None:
            U = undoScaling(sId, U, "uMean", "uStd", missing_marker=missing_marker)
    return zPred, yPred, xPred, Y, Z, U


def evalSysId(
    sId,
    YTest=None,
    ZTest=None,
    YTrain=None,
    ZTrain=None,
    UTest=None,
    UTrain=None,
    trueSys=None,
    YType=None,
    ZType=None,
    useXFilt=False,
    missing_marker=None,
    undo_scaling=False,
):
    """Evaluates a learned model based on predictions and also in terms of model parameters if the true model is known.

    Args:
        sId (PredictorModel): a model that implements a predict method.
        YTest (np.array, optional): input test data. Defaults to None.
        ZTest (np.array, optional): output test data. Defaults to None.
        YTrain (np.array, optional): input training data. Defaults to None.
        ZTrain (np.array, optional): output training data. Defaults to None.
        UTest (np.array, optional): external input test data. Defaults to None.
        UTrain (np.array, optional): external training test data. Defaults to None.
        trueSys (LSSM, optional): true model, if known in simulations. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        useXFilt (bool, optional): if true, will pass to predict if the
            model supports that argument (i.e. is an LSSM). Defaults to False.
        missing_marker (numpy value, optional): indicator of missing samples in data.
            Is used in performing and undoing preprocessing. Defaults to None.
        undo_scaling (bool, optional): if true, will apply the inverse scaling
            on predictions. Defaults to False.

    Returns:
        perf (dict): computed performance measures
        zPredTest (np.array): predicted Z
        yPredTest (np.array): predicted Y
        xPredTest (np.array): latent state X
    """
    perf = {}
    zPredTest, yPredTest, xPredTest = None, None, None

    if YTest is not None:
        zPredTest, yPredTest, xPredTest, YTest, ZTest, UTest = runPredict(
            sId,
            YTest,
            ZTest,
            UTest,
            YType,
            ZType,
            useXFilt,
            missing_marker,
            undo_scaling,
        )
        perfD = evaluateDecoding(
            ZTest=ZTest,
            zPredTest=zPredTest,
            YTest=YTest,
            yPredTest=yPredTest,
            YType=YType,
            ZType=ZType,
            missing_marker=missing_marker,
        )
        perf.update(perfD)

    return perf, zPredTest, yPredTest, xPredTest


def evaluateDecoding(
    ZTest=None,
    zPredTest=None,
    YTest=None,
    yPredTest=None,
    sId=None,
    missing_marker=None,
    YType=None,
    ZType=None,
    measures=["CC", "NRMSE", "EV", "R2", "AUC", "ACC", "ACCD1", "PoissonLL", "CM"],
):
    """Evaluates prediction of data

    Args:
        ZTest (np.array, optional): true values of the z data. Defaults to None.
        zPredTest (np.array, optional): predicted values of the z data. Defaults to None.
        YTest (np.array, optional): true values of the y data. Defaults to None.
        yPredTest (np.array, optional): predicted values of the y data. Defaults to None.
        sId (object, optional): learned model. Defaults to None.
        missing_marker (number, optional): the marker value for missing data. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        measures (list, optional): list of performance measures to compute when possible.
            Defaults to ['CC', 'NRMSE', 'EV', 'R2', 'AUC', 'ACC', 'ACCD1', 'PoissonLL', 'CM'].

    Returns:
        errs (dict): computed performance measures
    """
    if zPredTest is None and yPredTest is None and sId is not None:
        zPredTest, yPredTest, xPredTest = sId.predict(YTest)

    if zPredTest is not None and ZTest is not None:
        nonTAx = np.arange(1, len(zPredTest.shape))
        nonTAxForTrue = nonTAx[nonTAx < len(ZTest.shape)]
        zNotBlown = np.all(
            np.logical_not(np.logical_or(np.isnan(zPredTest), np.isinf(zPredTest))),
            axis=tuple(nonTAx),
        )
        zNotNaN = np.all(np.logical_not(np.isnan(ZTest)), axis=tuple(nonTAxForTrue))
        if missing_marker is not None:
            zNotMissing = np.all(ZTest != missing_marker, axis=1)
        else:
            zNotMissing = zNotBlown
        zOk = np.nonzero(
            np.logical_and(zNotNaN, np.logical_and(zNotBlown, zNotMissing))
        )[0]

    if yPredTest is not None:
        nonTAx = np.arange(1, len(yPredTest.shape))
        nonTAxForTrue = nonTAx[nonTAx < len(YTest.shape)]
        yNotBlown = np.all(
            np.logical_not(np.logical_or(np.isnan(yPredTest), np.isinf(yPredTest))),
            axis=tuple(nonTAx),
        )
        yNotNaN = np.all(np.logical_not(np.isnan(YTest)), axis=tuple(nonTAxForTrue))
        if missing_marker is not None:
            yNotMissing = np.all(YTest != missing_marker, axis=1)
        else:
            yNotMissing = yNotBlown
        yOk = np.nonzero(
            np.logical_and(yNotNaN, np.logical_and(yNotBlown, yNotMissing))
        )[0]

    errs = {}

    for m in measures:
        if zPredTest is not None and ZTest is not None:
            if (
                len(zPredTest.shape) == 2
                and m not in ["AUC", "ACC", "ACCD1", "CM"]
                and (m != "PoissonLL" or ZType == "count_process")
            ):
                errs[m] = evalPrediction(ZTest[zOk, :], zPredTest[zOk, :], m)
                errs["mean" + m] = np.mean(errs[m])
            elif len(zPredTest.shape) == 3 and m in ["AUC", "ACC", "ACCD1", "CM"]:
                errs[m] = evalPrediction(ZTest[zOk, :], zPredTest[zOk, :, :], m)
                if m == "CM":
                    if zOk.size == 0:
                        errs[m] = (
                            np.ones(
                                (
                                    zPredTest.shape[2],
                                    zPredTest.shape[2],
                                    zPredTest.shape[1],
                                )
                            )
                            * np.nan
                        )
                    errs["mean" + m] = np.mean(errs[m], axis=2)
                else:
                    errs["mean" + m] = np.mean(errs[m])

        if yPredTest is not None:
            if (
                len(yPredTest.shape) == 2
                and m not in ["AUC", "ACC", "ACCD1", "CM"]
                and (m != "PoissonLL" or YType == "count_process")
            ):
                errs["y" + m] = evalPrediction(YTest[yOk, :], yPredTest[yOk, :], m)
                errs["meany" + m] = np.mean(errs["y" + m])

    return errs


def evalPrediction(trueValue, prediction, measure, missing_marker=None):
    """Evaluates prediction of data

    Args:
        trueValue (np.array): true values. The first dimension is taken as the sample dimension
            over which metrics are computed.
        prediction (np.array): predicted values
        measure (string): performance measure name
        missing_marker: if not None, will ignore samples with this value (default: None)

    Returns:
        perf (np.array): the value of performance measure, computed for each dimension of data
    """
    if missing_marker is not None:
        if np.isnan(missing_marker):
            isOk = np.all(~np.isnan(prediction), axis=1)
        else:
            isOk = np.all(prediction != missing_marker, axis=1)
        trueValue = copy.deepcopy(trueValue)[isOk, :]
        prediction = copy.deepcopy(prediction)[isOk, :]
    nSamples, nDims = trueValue.shape
    if nSamples == 0:
        perf = np.nan * np.ones((nDims,))
        return perf
    isFlat = (np.max(trueValue, axis=0) - np.min(trueValue, axis=0)) == 0
    if measure == "CC":
        if nSamples < 2:
            perf = np.nan * np.ones((nDims,))
            return perf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = np.corrcoef(trueValue, prediction, rowvar=False)
        perf = np.diag(R[nDims:, :nDims])
        if np.any(
            isFlat
        ):  # This is the fall-back values for flat true signals in the corrcoef code, but it may not detect flats correctly, so we help it
            perf.setflags(write=1)
            perf[isFlat] = np.nan
    elif measure == "MSE":
        perf = metrics.mean_squared_error(
            trueValue, prediction, multioutput="raw_values"
        )
    elif measure == "RMSE":
        MSE = evalPrediction(trueValue, prediction, "MSE")
        perf = np.sqrt(MSE)
    elif measure == "NRMSE":
        RMSE = evalPrediction(trueValue, prediction, "RMSE")
        std = np.std(trueValue, axis=0)
        perf = np.empty(RMSE.size)
        perf[:] = np.nan
        perf[~isFlat] = RMSE[~isFlat] / std[~isFlat]
    elif measure == "MAE":
        perf = metrics.mean_absolute_error(
            trueValue, prediction, multioutput="raw_values"
        )
    elif measure == "NMAE":
        MAE = evalPrediction(trueValue, prediction, "MAE")
        denom = metrics.mean_absolute_error(
            trueValue - np.mean(trueValue, axis=0),
            np.zeros_like(prediction),
            multioutput="raw_values",
        )
        perf = np.empty(MAE.size)
        perf[:] = np.nan
        perf[~isFlat] = MAE[~isFlat] / denom[~isFlat]
    elif measure == "EV":
        perf = metrics.explained_variance_score(
            trueValue, prediction, multioutput="raw_values"
        )
        perf[isFlat] = (
            0  # This is the fall-back values for flat true signals in the explained_variance_score code, but it may not detect flats correctly, so we help it
        )
    elif measure == "R2":
        if nSamples < 2:
            perf = np.nan * np.ones((nDims,))
            return perf
        perf = metrics.r2_score(trueValue, prediction, multioutput="raw_values")
        perf[isFlat] = (
            0  # This is the fall-back values for flat true signals in the r2_score code, but it may not detect flats correctly, so we help it
        )
    # Classification measures
    elif measure == "ACC":
        if len(prediction.shape) == 3:  # Probabilities are given
            prediction = np.argmax(prediction, axis=2)
        perf = np.empty(trueValue.shape[1])
        for yi in range(trueValue.shape[1]):
            perf[yi] = metrics.accuracy_score(
                trueValue[:, yi], prediction[:, yi], normalize=True
            )
    elif measure[:-1] == "ACCD" and measure[-1] in [str(d) for d in range(10)]:
        okDist = float(measure[-1])
        if len(prediction.shape) == 3:  # Probabilities are given
            prediction = np.argmax(prediction, axis=2)
        perf = np.empty(trueValue.shape[1])
        for yi in range(trueValue.shape[1]):
            thisT = trueValue[:, yi]
            thisP = np.array(prediction[:, yi])
            isClose = np.abs(thisP - thisT) <= okDist
            thisP[isClose] = thisT[isClose]
            perf[yi] = metrics.accuracy_score(thisT, thisP, normalize=True)
    elif measure == "AUC":
        perf = np.empty(trueValue.shape[1])
        for yi in range(trueValue.shape[1]):
            try:
                if prediction.shape[-1] > 2:  # multiclass
                    perf[yi] = metrics.roc_auc_score(
                        trueValue[:, yi],
                        prediction[:, yi, :],
                        average="macro",
                        multi_class="ovo",
                    )
                else:  # Binary
                    perf[yi] = metrics.roc_auc_score(
                        trueValue[:, yi],
                        prediction[:, yi, -1],
                        average="macro",
                        multi_class="ovo",
                    )
            except Exception as e:
                perf[yi] = np.nan
                print('Error: "{}". Will take auc to be nan'.format(e))
    elif measure == "CM":
        NClass = prediction.shape[2]
        perf = np.empty((NClass, NClass, trueValue.shape[1]))
        for yi in range(trueValue.shape[1]):
            y_pred = np.argmax(prediction[:, yi, :], axis=1)
            perf[:, :, yi] = metrics.confusion_matrix(
                trueValue[:, yi], y_pred, labels=np.arange(NClass)
            )
    elif measure == "PoissonLL":
        perf = np.nanmean(
            prediction - trueValue * np.log(prediction), axis=0
        )  # See https://www.tensorflow.org/api_docs/python/tf/keras/losses/poisson
    return perf


def isHigherBetter(perfMeasure):
    if "MSE" in perfMeasure or "Time" in perfMeasure:
        return False
    else:
        return True


def isWithin1SEMOfTheBest(yVals, yValsCVSEM, perfField, peak_det_sem_multiplier=0):
    """Find smallest dim within 1 sem of the peak performance

    Args:
        yVals (_type_): _description_
        yValsCVSEM (_type_): _description_
        perfField (_type_): _description_
        peak_det_sem_multiplier (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if isHigherBetter(perfField):
        peakPerfMinusSEM = np.nanmax(
            yVals - yValsCVSEM * peak_det_sem_multiplier, axis=0
        )[np.newaxis, :]
        isOk = yVals >= peakPerfMinusSEM
    else:
        peakPerfPlusSEM = np.nanmin(
            yVals + yValsCVSEM * peak_det_sem_multiplier, axis=0
        )[np.newaxis, :]
        isOk = yVals <= peakPerfPlusSEM
    return isOk


def isWithinRatioOfTheBest(yVals, perfField, ratio=0):
    """Find smallest dim within ratio*best of the best. For example, if ratio is 0.05, we find smallest
    dim that's within 5% of the best.

    Args:
        yVals (_type_): _description_
        perfField (_type_): _description_
        ratio (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if isHigherBetter(perfField):
        peakPerfMinusRatio = np.nanmax(yVals - ratio * np.abs(yVals), axis=0)[
            np.newaxis, :
        ]
        isOk = yVals >= peakPerfMinusRatio
    else:
        peakPerfPlusRatio = np.nanmin(yVals + ratio * np.abs(yVals), axis=0)[
            np.newaxis, :
        ]
        isOk = yVals <= peakPerfPlusRatio
    return isOk


def isAlmostAsGoodAsTheBest(
    yVals,
    yValsCVSEM,
    perfField,
    criteria="within_sem",
    peak_det_sem_multiplier=0,
    ratio=0,
):
    """Checks if result is as close as the peak performing result, according to the specified criteria

    Args:
        yVals (_type_): _description_
        yValsCVSEM (_type_): _description_
        perfField (_type_): _description_
        criteria (str, optional): _description_. Defaults to 'within_sem'.
        peak_det_sem_multiplier (int, optional): _description_. Defaults to 0.
        ratio (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if criteria == "within_sem":
        return isWithin1SEMOfTheBest(
            yVals,
            yValsCVSEM,
            perfField,
            peak_det_sem_multiplier=peak_det_sem_multiplier,
        )
    elif criteria == "within_ratio":
        return isWithinRatioOfTheBest(yVals, perfField, ratio=ratio)
    else:
        raise (Exception(f'Criteria "{criteria}" is not supported'))


def getCVedPerf(
    yValsF,
    perfField,
    xValAxis=[0],
    preMeanAxes=[],
    peak_det_criteria="within_sem",
    peak_det_sem_multiplier=1,
    peak_det_ratio=0.05,
    yValsF2=None,
    perfField2=None,
    verbose=False,
    findBestIndsAll=True,
):
    """Prepares cross-validated performance by averaging over folds, returns the SEM,
    and the smallest dimension that reaches close to the peak performance

    Args:
        yValsF (_type_): _description_
        perfField (_type_): _description_
        xValAxis (list, optional): _description_. Defaults to [0].
        preMeanAxes (list, optional): _description_. Defaults to [].
        peak_det_criteria (str, optional): _description_. Defaults to 'within_sem'.
        peak_det_sem_multiplier (int, optional): _description_. Defaults to 1.
        peak_det_ratio (float, optional): _description_. Defaults to 0.05.
        yValsF2 (_type_, optional): _description_. Defaults to None.
        perfField2 (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
    """

    def prepYValsAndSEM(yValsF):
        yValsPM = np.array(yValsF)
        yValsPMSEM = np.zeros_like(yValsPM)
        for pmAx in preMeanAxes:  # Taking mean over these axes doesn't show up in SEM
            yValsPM = np.nanmean(yValsF, axis=pmAx)
            yValsPM = np.expand_dims(
                yValsPM, axis=pmAx
            )  # To keep axes in their original place
            yValsPMSEM = np.nanstd(yValsF, axis=pmAx) / np.sqrt(yValsF.shape[pmAx])
            yValsPMSEM = np.expand_dims(
                yValsPMSEM, axis=pmAx
            )  # To keep axes in their original place

        nonXAxes = np.where(
            np.logical_not(np.isin(range(len(yValsPM.shape)), xValAxis))
        )[
            0
        ]  # Mean over these axes will be included in SEM
        yVals = np.transpose(
            yValsPM, tuple(np.concatenate((xValAxis, nonXAxes)))
        )  # Bring the dimension that is not going to be averaged to the front
        yVals = np.reshape(yVals, (yVals.shape[0], np.prod(yVals.shape[1:])), order="F")
        yValsCVSEM = np.transpose(
            yValsPMSEM, tuple(np.concatenate((xValAxis, nonXAxes)))
        )  # Bring the dimension that is not going to be averaged to the front
        yValsCVSEM = np.reshape(
            yValsCVSEM, (yValsCVSEM.shape[0], np.prod(yValsCVSEM.shape[1:])), order="F"
        )
        return yVals, yValsCVSEM

    yVals, yValsCVSEM = prepYValsAndSEM(yValsF)
    isOk = isAlmostAsGoodAsTheBest(
        yVals,
        yValsCVSEM,
        perfField,
        criteria=peak_det_criteria,
        peak_det_sem_multiplier=peak_det_sem_multiplier,
        ratio=peak_det_ratio,
    )
    if verbose:
        logger.info(
            "Of {} cases, the following were within {} sem of the best {} in each session:\n{}".format(
                isOk.shape[0], peak_det_sem_multiplier, perfField, np.where(isOk)[0]
            )
        )
    if yValsF2 is not None:
        yVals2, yValsCVSEM2 = prepYValsAndSEM(yValsF2)
        isOk2 = isAlmostAsGoodAsTheBest(
            yVals2,
            yValsCVSEM2,
            perfField2,
            criteria=peak_det_criteria,
            peak_det_sem_multiplier=peak_det_sem_multiplier,
            ratio=peak_det_ratio,
        )
        if verbose:
            logger.info(
                "Criteria 2: {}\nOf {} cases, the following were within {} sem of the best {} in each session:\n{}".format(
                    perfField2,
                    isOk2.shape[0],
                    peak_det_sem_multiplier,
                    perfField2,
                    np.where(isOk2)[0],
                )
            )
        if isHigherBetter(perfField2):
            criteria = yVals2
        else:
            criteria = -yVals2  # Because below we assume higher "criteria" is better
        hasShared = (
            np.ones((isOk.shape[0], 1), bool)
            * np.any(np.logical_and(isOk, isOk2), axis=0)[np.newaxis]
        )
        criteria[np.logical_and(~isOk2, hasShared)] = (
            np.nan
        )  # If there are some cases that are simultaneously within 1sem of both perf measures, only pick among those
        if verbose:
            logger.info(
                "- In {}/{} sessions ({}) some cases satisfied both criteria, will pick among those.".format(
                    np.sum(hasShared[0, :]),
                    hasShared.shape[-1],
                    np.where(hasShared[0, :])[0],
                )
            )
            logger.info(
                "- In {}/{} sessions ({}), no case satisfied both criteria, so will pick the case with best criteria 2 ({}) among those that satisfy criteria 1 ({}).".format(
                    np.sum(~hasShared[0, :]),
                    hasShared.shape[-1],
                    np.where(~hasShared[0, :])[0],
                    perfField2,
                    perfField,
                )
            )
    else:
        # By default, criteria will be the dimension index in the first axis (usually corresponds to nx in the results)
        # Negative sign is because below we assume higher "criteria" is better
        criteria = (
            -np.arange(yVals.shape[0])[:, np.newaxis]
            * np.ones(yVals.shape[1])[np.newaxis, :]
        )
        if verbose:
            logger.info(
                "- Will pick the first case (smallest index) that among those that satisfy criteria 1 ({}).".format(
                    perfField
                )
            )
    # Among those that are ok (isOk == True), pick the one with the smallest "criteria" value
    criteria[~isOk] = (
        np.nan
    )  # At least one will be ok (the case that had the best result)
    peakPerfXValInds = np.argmax(
        criteria, axis=0
    )  # argmax will stop at the first repetition of the max value in criteria (could pick nan)
    notPureNan = np.any(~np.isnan(criteria), axis=0)
    peakPerfXValInds[notPureNan] = np.nanargmax(
        criteria[:, notPureNan], axis=0
    )  # nanargmax will stop at the first non-nan repetition of the max value in criteria
    # Let's also repeat this until we have a sorted list of indices based on their preferance
    bestIndsAll = []
    try:
        for mi in range(len(peakPerfXValInds)):
            remInds = np.array(range(0, yValsF.shape[xValAxis[0]]))
            bestIndsAllThis = []
            if findBestIndsAll:
                while len(remInds) > 0 or len(bestIndsAllThis) == 0:
                    if len(bestIndsAllThis) == 0:
                        thisBestInd = peakPerfXValInds
                    else:
                        thisBestIndInRem = getCVedPerf(
                            yValsF[remInds, ...],
                            perfField,
                            xValAxis=xValAxis,
                            preMeanAxes=preMeanAxes,
                            peak_det_criteria=peak_det_criteria,
                            peak_det_sem_multiplier=peak_det_sem_multiplier,
                            peak_det_ratio=peak_det_ratio,
                            yValsF2=(
                                yValsF2[remInds, ...] if yValsF2 is not None else None
                            ),
                            perfField2=perfField2,
                            verbose=False,
                            findBestIndsAll=False,
                        )[2]
                        thisBestInd = remInds[thisBestIndInRem]
                    bestIndsAllThis.append(thisBestInd[mi])
                    remInds = np.delete(remInds, remInds == thisBestInd[mi])
            bestIndsAll.append(bestIndsAllThis)
    except Exception as e:
        # logger.warning(f'Complete sort by perf failed (only needed for ensembling), err: {e}')
        pass
    return yVals, yValsCVSEM, peakPerfXValInds, bestIndsAll


def computePairwiseStatsTests(yValsA):
    """Computes pairwise statitical tests between multiple sets of results

    Args:
        yValsA (_type_): _description_

    Returns:
        _type_: _description_
    """
    pVals = np.empty((len(yValsA), len(yValsA)))
    pVals[:] = np.nan
    pValNs = np.empty((len(yValsA), len(yValsA), 2))
    pVals[:] = np.nan
    for ind1 in range(len(yValsA)):
        for ind2 in range(len(yValsA)):
            y0, y1 = yValsA[ind1], yValsA[ind2]
            isOk = np.logical_and(~np.isnan(y1), ~np.isnan(y0))
            if ind1 == ind2 or np.all(y0 == y1):
                p = 1
            elif np.any(isOk):
                try:
                    w, p = wilcoxon(y1[isOk], y0[isOk], alternative="greater")
                except Exception as e:
                    logger.info("Wilcoxon error: {}".format(e))
            pVals[ind1, ind2] = p
            pValNs[ind1, ind2, :] = (y0[isOk].size, y1[isOk].size)
    return pVals, pValNs


def computeMaskedStats(yVals, yValsBad=None, axis=1):
    """Computes mean, median, Std and SEM stats for data that has masked samples

    Args:
        yVals (_type_): _description_
        yValsBad (_type_, optional): _description_. Defaults to None.
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if yValsBad is None:
        yValsBad = np.logical_or(np.isnan(yVals), np.isinf(yVals))
    yValsMA = np.ma.array(yVals, mask=yValsBad)
    yValsMean = np.ma.mean(yValsMA, axis=axis)
    yValsMedian = np.ma.median(yValsMA, axis=axis)
    yValsStd = np.ma.std(yValsMA, axis=axis)
    yValsSEM = yValsStd / np.sqrt(np.ma.count(yValsMA, axis=axis))
    return yValsMean, yValsMedian, yValsStd, yValsSEM


def findPerformanceFrontier(
    perfVals, pairwisePVals, perfNames, min_relative_diff=0, labels=None, verbose=False
):
    """Finds the performance frontier for a set of metrics.

    Args:
        perfVals (_type_): _description_
        pairwisePVals (_type_): _description_
        perfNames (_type_): _description_
        min_relative_diff (int, optional): _description_. Defaults to 0.
        labels (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # plt.figure(); plotPairwisePValueMap(pairwisePVals[0], text_args={'fontsize': 10}, yticklabels=labels); plt.title(perfNames[0])
    # plt.figure(); plotPairwisePValueMap(pairwisePVals[1], text_args={'fontsize': 10}, yticklabels=labels); plt.title(perfNames[1])
    onFrontier = np.zeros(pairwisePVals[0].shape[0], dtype=bool)
    for mi in range(pairwisePVals[0].shape[0]):
        isOnTheFrontier = True
        for mi2 in range(pairwisePVals[0].shape[0]):
            # Compare mi with mi2, to see if mi is better in some metric or is similar in all metrics
            isWorse = False
            isBetter = False
            for perfName, pVals, perfValsThis in zip(
                perfNames, pairwisePVals, perfVals
            ):
                perfValsMean = np.mean(perfValsThis, axis=-1)
                perfRelDiff = np.abs(perfValsMean[mi] - perfValsMean[mi2]) / np.abs(
                    perfValsMean[mi2]
                )
                if perfRelDiff < min_relative_diff:
                    continue
                if (
                    isHigherBetter(perfName) and pVals[mi2, mi] <= 0.05
                ):  # Higher is better and m1 is better than m2
                    isBetter = True
                elif (
                    not isHigherBetter(perfName) and pVals[mi, mi2] <= 0.05
                ):  # Lower is better and m1 is better than m2
                    isBetter = True
                elif (
                    isHigherBetter(perfName) and pVals[mi, mi2] <= 0.05
                ):  # Higher is better and m1 is worse than m2
                    isWorse = True
                elif (
                    not isHigherBetter(perfName) and pVals[mi2, mi] <= 0.05
                ):  # Lower is better and m1 is worse than m2
                    isWorse = True
            if not isBetter and isWorse:
                isOnTheFrontier = False
                if verbose and labels is not None:
                    logger.info(
                        "{} not on frontier because compared with {} it has no benefit".format(
                            labels[mi], labels[mi2]
                        )
                    )
                break
        onFrontier[mi] = isOnTheFrontier
    return onFrontier


def fetchField(x, fieldName, meanElems):
    if type(x) is dict and fieldName in x:
        if meanElems is None:
            return x[fieldName]
        elif meanElems == True:
            return np.nanmean(x[fieldName])
        else:
            return np.nanmean(x[fieldName][meanElems])
    else:
        return np.nan


def getPerfVals(
    perfVals,
    perfField,
    xValField=None,
    xValAxis=[0],
    xVals=None,
    meanElems=None,
    groupAxis=None,
    preMeanAxes=None,
    nxSelField=None,
    peak_det_criteria="within_sem",
    peak_det_sem_multiplier=1,
    peak_det_ratio=0.05,
):
    """Extracts cross-validated performances and the min dimensions required to reach close to them

    Args:
        perfVals (_type_): _description_
        perfField (_type_): _description_
        xValField (_type_, optional): _description_. Defaults to None.
        xValAxis (list, optional): _description_. Defaults to [0].
        xVals (_type_, optional): _description_. Defaults to None.
        meanElems (_type_, optional): _description_. Defaults to None.
        groupAxis (_type_, optional): _description_. Defaults to None.
        preMeanAxes (_type_, optional): _description_. Defaults to None.
        nxSelField (_type_, optional): _description_. Defaults to None.
        peak_det_criteria (str, optional): _description_. Defaults to 'within_sem'.
        peak_det_sem_multiplier (int, optional): _description_. Defaults to 1.
        peak_det_ratio (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    if xVals is None:
        xVals = []
    if isinstance(xVals, list):
        xVals = np.array(xVals)
    yValsA = np.vectorize(fetchField)(perfVals, perfField, meanElems)
    if groupAxis is None:
        groupAxis = []
    if len(groupAxis) == 0:
        groupAxis = [0]
        groupInds = [list(range(yValsA.shape[0]))]
    else:
        groupAxis = copy.deepcopy(groupAxis)
        groupInds = [
            [i] for i in range(yValsA.shape[groupAxis[0]])
        ]  # Plot this axis one by one
    if preMeanAxes is None:
        preMeanAxes = []
    if nxSelField is None:
        nxSelField = perfField
    if not isinstance(nxSelField, (list, tuple)):
        nxSelField = [nxSelField]
    yValsANxSel = [
        np.vectorize(fetchField)(perfVals, nxSelFld, meanElems)
        for nxSelFld in nxSelField
    ]
    nxSelLabel = ""
    if peak_det_criteria == "within_sem":
        nxSelLabel += f"within {peak_det_sem_multiplier}sem "
    elif peak_det_criteria == "within_ratio":
        nxSelLabel += f"within {peak_det_ratio*100}% "
    nxSelLabel += "peak " + "^".join(nxSelField)
    res = {
        "perfField": perfField,
        "nxSelField": nxSelField,
        "nxSelLabel": nxSelLabel,
        "groupAxis": groupAxis,
        "groupInds": groupInds,
        "xValsAllPreMean": [],
        "xValsAll": [],
        "yValsAll": [],
        "peakPerfXValIndsAll": [],
        "peakPerfXValIndsOrigShapeAll": [],
        "peakPerfXValAll": [],
        "peakPerfYValAll": [],
        "peakPerfOfM1YValAll": [],
        "peak_det_sem_multiplier": peak_det_sem_multiplier,
    }
    for li, groupInd in enumerate(groupInds):
        yValsF = np.take(
            yValsA, groupInd, axis=groupAxis[0]
        )  # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].
        peakPerfXValIndsAll = []
        for nxSelFldInd, nxSelFld in enumerate(nxSelField):
            yValsFNxSel = np.take(
                yValsANxSel[nxSelFldInd], groupInd, axis=groupAxis[0]
            )  # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].

            (
                nxSelFldVals,
                nxSelFldValsCVSEM,
                peakPerfXValIndsThis,
                bestIndsAll,
            ) = getCVedPerf(
                yValsFNxSel,
                nxSelFld,
                xValAxis=xValAxis,
                preMeanAxes=preMeanAxes,
                peak_det_criteria=peak_det_criteria,
                peak_det_sem_multiplier=peak_det_sem_multiplier,
                peak_det_ratio=peak_det_ratio,
            )
            peakPerfXValIndsAll.append(peakPerfXValIndsThis)

        peakPerfXValIndsAll = np.array(peakPerfXValIndsAll)
        peakPerfXValInds = np.max(peakPerfXValIndsAll, axis=0)

        yVals, yValsCVSEM, yValsPeakPerfXValInds, bestIndsAll = getCVedPerf(
            yValsF,
            perfField,
            xValAxis=xValAxis,
            preMeanAxes=preMeanAxes,
            peak_det_criteria=peak_det_criteria,
            peak_det_sem_multiplier=peak_det_sem_multiplier,
            peak_det_ratio=peak_det_ratio,
        )

        nonXAxes = np.where(
            np.logical_not(np.isin(range(len(yValsF.shape)), xValAxis))
        )[
            0
        ]  # Mean over these axes will be included in SEM

        # Find peakPerfXValInds in the original shape
        def recover(input_arr):
            if len(preMeanAxes) > 0:
                input_arr_tiled = np.tile(
                    input_arr, np.prod(np.array(yValsF.shape)[preMeanAxes])
                )
            else:
                input_arr_tiled = input_arr
            yValsRecov = np.reshape(
                input_arr_tiled,
                np.array(yValsF.shape)[np.concatenate((xValAxis, nonXAxes))],
                order="F",
            )
            yValsFRecov = np.transpose(
                yValsRecov, np.insert(np.arange(1, len(yValsF.shape)), xValAxis, 0)
            )
            if len(preMeanAxes) > 0:
                np.testing.assert_equal(
                    np.max(yValsFRecov, axis=preMeanAxes[0]),
                    np.min(yValsFRecov, axis=preMeanAxes[0]),
                )  # Make sure the same value is repeated for all preMeanAxes
            return yValsFRecov

        if len(preMeanAxes) == 0:
            yValsFRecov = recover(yVals)
            np.testing.assert_equal(
                yValsFRecov, yValsF
            )  # Make sure the recovery function reorganizes yVals correctly to match original yValsF
        peakPerfXValIndsOrigShape = recover(
            np.ones(np.array(yValsF.shape)[xValAxis])[:, np.newaxis]
            @ peakPerfXValInds[np.newaxis, :]
        )

        # Find xValsThis
        if len(xVals) > 0:
            xValsThis = xVals
            if isinstance(xValsThis[0], (list, tuple, np.ndarray)):
                xValsThis = xVals[li]
            xValsAll = xValsThis[:, np.newaxis]
        elif xValField is not None:
            xValsA = np.vectorize(fetchField)(perfVals, xValField, meanElems)
            xValsF = np.take(
                xValsA, groupInd, axis=groupAxis[0]
            )  # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].
            xValsAll, xValsCVSEM, peakPerfXValIndsCopy, bestIndsAll = getCVedPerf(
                xValsF,
                xValField,
                xValAxis=xValAxis,
                preMeanAxes=preMeanAxes,
                peak_det_criteria=peak_det_criteria,
                peak_det_sem_multiplier=peak_det_sem_multiplier,
                peak_det_ratio=peak_det_ratio,
            )
            xValsThis = np.nanmean(xValsAll, axis=1)

        res["xValsAll"].append(xValsThis)
        res["xValsAllPreMean"].append(xValsAll)
        res["yValsAll"].append(yVals)

        res["peakPerfXValIndsAll"].append(peakPerfXValInds)
        res["peakPerfXValIndsOrigShapeAll"].append(peakPerfXValIndsOrigShape)
        # res['peakPerfXValAll'].append( np.array(xValsThis)[peakPerfXValInds] )
        res["peakPerfXValAll"].append(
            np.take_along_axis(xValsAll, peakPerfXValInds[np.newaxis, :], axis=0)[0, :]
        )
        res["peakPerfYValAll"].append(
            np.take_along_axis(yVals, peakPerfXValInds[np.newaxis, :], axis=0)[0, :]
        )
        res["peakPerfOfM1YValAll"].append(
            np.take_along_axis(
                yVals, res["peakPerfXValIndsAll"][0][np.newaxis, :], axis=0
            )[0, :]
        )
    return res


def sanitize_perfs(perfs, use_old_perf_sanitization=False, zSource=None):
    """Removes some basic issues from an array of performance metric dicts

    Args:
        perfs (array of dicts): array of performance measure dicts

    Returns:
        perfs: sanitized perfs
        upCnt: number of updated elements
    """
    if not isinstance(perfs, np.ndarray):
        perfs = np.array([perfs])[np.newaxis, np.newaxis, np.newaxis, :]
    if perfs.size == 0:
        return perfs, 0
    if "yCC" in perfs.flatten()[0]:
        numDims = len(perfs.shape)
        flatYDimsCnt = []
        if numDims == 4:
            sessRng = range(perfs.shape[0])
            methodRng = range(perfs.shape[1])
        else:
            sessRng = range(1)
            methodRng = range(perfs.shape[0])
        for si in sessRng:
            sessPerf = perfs[si, ...] if numDims == 4 else perfs
            yCCField = "yCC"
            # if yCCField not in sessPerf.flatten()[0].keys():
            #     yCCField = [k for k in sessPerf.flatten()[0].keys() if 'yCC' in k and 'mean' not in k][0]
            nyVals = np.vectorize(
                lambda sp: sp[yCCField].size if yCCField in sp else 0
            )(sessPerf)
            nyU = np.unique(nyVals)
            if len(nyU) > 1:
                logger.warning(
                    "ny is not consistent across methods (probably aggregating different features), skipping yCC sanitize_perfs for sessions {}/{}".format(
                        1 + si, len(sessRng)
                    )
                )
                continue
            for yi in range(nyU[0]):
                yCCThis = np.vectorize(lambda a: a["yCC"][yi])(sessPerf)[
                    np.newaxis, ...
                ]
                yR2This = np.vectorize(lambda a: a["yR2"][yi])(sessPerf)[
                    np.newaxis, ...
                ]
                if yi == 0:
                    yCC = np.copy(yCCThis)
                    yR2 = np.copy(yR2This)
                else:
                    yCC = np.concatenate((yCC, yCCThis), axis=0)
                    yR2 = np.concatenate((yR2, yR2This), axis=0)
            for fi in range(yCC.shape[2]):
                yCCFold = yCC[:, :, fi, :]
                yR2Fold = yR2[:, :, fi, :]
                if use_old_perf_sanitization:
                    yCCFoldFlat = np.reshape(
                        yCCFold,
                        (yCCFold.shape[0], yCCFold.shape[1] * yCCFold.shape[2]),
                        "F",
                    )
                    isFlat = np.all(
                        np.logical_or(np.isnan(yCCFoldFlat), yCCFoldFlat < 1e-10),
                        axis=1,
                    )
                else:
                    yR2FoldFlat = np.reshape(
                        yR2Fold,
                        (yR2Fold.shape[0], yR2Fold.shape[1] * yR2Fold.shape[2]),
                        "F",
                    )
                    isFlat = np.all(
                        np.logical_or(np.isnan(yR2FoldFlat), yR2FoldFlat < -1e10),
                        axis=1,
                    )  # This makes more sense for detecting flat channels
                if np.any(isFlat):
                    isFlatInds = np.where(isFlat)[0]
                    for mi in methodRng:
                        for nxi in range(perfs.shape[-1]):
                            if numDims == 4:
                                inds = (si, mi, fi, nxi)
                            else:
                                inds = (mi, fi, nxi)
                            if not perfs[inds]["yCC"].flags.writeable:
                                perfs[inds]["yCC"].flags.writeable = True
                                perfs[inds]["yR2"].flags.writeable = True
                                perfs[inds]["yEV"].flags.writeable = True
                            perfs[inds]["yCC"][isFlat] = (
                                np.nan if not use_old_perf_sanitization else 0
                            )
                            perfs[inds]["yR2"][isFlat] = (
                                np.nan if not use_old_perf_sanitization else 0
                            )
                            perfs[inds]["yEV"][isFlat] = (
                                np.nan if not use_old_perf_sanitization else 0
                            )
                flatYDimsCnt.append(np.sum(isFlat))
        flatYDimsCnt = np.array(flatYDimsCnt)
        if np.sum(flatYDimsCnt > 0):
            print(
                "Detected {}/{} folds with an average of {:.3g} flat y dimensions. yCC for all was set to nan.".format(
                    np.sum(flatYDimsCnt > 0),
                    flatYDimsCnt.size,
                    np.mean(flatYDimsCnt[flatYDimsCnt > 0]),
                )
            )

    upCnt = np.zeros_like(perfs)
    iterator = np.nditer(perfs, flags=["refs_ok", "multi_index"])
    while not iterator.finished:
        i = iterator.multi_index
        is_not_finished = iterator.iternext()
        perf = perfs[i]
        # Sanitize R2 and EV values for flat test sets
        for perfName in list(dict(perf).keys()):
            meanFieldName = "mean" + perfName
            if (
                ("R2" in perfName or "EV" in perfName or "CC" in perfName)
                and "mean" not in perfName
                and (
                    np.any(perf[perfName] < -1e10)
                    or np.any(np.isnan(perf[perfName]))
                    or (meanFieldName in perf and np.isnan(perf[meanFieldName]))
                )
            ):
                if np.any(perf[perfName] < -1e10):
                    perf[perfName][perf[perfName] < -1e10] = np.nan
                if meanFieldName in perf:
                    perf[meanFieldName] = np.nanmean(perf[perfName])
                upCnt[i] += 1
        # Add SNR measures
        perfKeys = list(dict(perf).keys())
        if "R2" in perfKeys:

            def R2ToSNR(R2):
                return -10 * np.log10(1 - R2)

            perf["SNR"] = R2ToSNR(perf["R2"])
            perf["meanSNR"] = np.mean(perf["SNR"])
        if zSource is not None and "label" in zSource[0]:
            # Add pos only and vel only decoding performances
            for perfName in ["R2", "CC"]:
                meanFieldName = "mean" + perfName
                for z_subset in ["vel", "pos"]:
                    z_inds = [
                        zi for zi, zs in enumerate(zSource) if z_subset in zs["label"]
                    ]
                    if len(z_inds) > 0 and len(z_inds) < len(zSource):
                        perf[perfName + z_subset] = perf[perfName][z_inds]
                        perf[meanFieldName + z_subset] = np.mean(perf[perfName][z_inds])
    return perfs, upCnt
