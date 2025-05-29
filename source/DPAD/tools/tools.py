""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Tools for system identification"""

import copy
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


def extractDiagonalBlocks(A, emptySide="both", absThr=np.spacing(1)):
    if emptySide == "either":
        BLKSU = extractDiagonalBlocks(A, "upper", absThr)
        BLKSL = extractDiagonalBlocks(A, "lower", absThr)
        if len(BLKSU) >= len(BLKSL):
            BLKS = BLKSU
        else:
            BLKS = BLKSL
        return BLKS

    j = 0
    BLKS = np.empty(0, dtype=int)
    while j < A.shape[0]:
        if emptySide == "both" or emptySide == "upper":
            for j1 in range(j, A.shape[1]):
                if j1 == (A.shape[1] - 1) or np.all(
                    np.abs(A[j : (j1 + 1), (j1 + 1) :]) <= absThr
                ):
                    i1 = j1 - j + 1
                    break

        if emptySide == "both" or emptySide == "lower":
            for j2 in range(j, A.shape[0]):
                if j2 == (A.shape[0] - 1) or np.all(
                    np.abs(A[(j2 + 1) :, j : (j2 + 1)]) <= absThr
                ):
                    i2 = j2 - j + 1
                    break

        if emptySide == "upper":
            i2 = i1
        elif emptySide == "lower":
            i1 = i2

        i = j + int(np.max([i1, i2]))
        BLKS = np.concatenate((BLKS, [(i - j)]))
        j = i

    return BLKS


def getBlockIndsFromBLKSArray(BLKS):
    if len(BLKS) == 0:
        return np.empty(0, dtype=int)

    BLKSCUM = np.array(np.atleast_2d(np.cumsum(BLKS)).T, dtype=int)
    groups = np.concatenate(
        (
            np.concatenate((np.zeros((1, 1), dtype=int), BLKSCUM[:-1, :]), axis=0),
            BLKSCUM,
        ),
        axis=1,
    )

    return groups


def applyFuncIf(Y, func):
    """Applies a function on Y itself if Y is an array or on each element of Y if it is a list/tuple of arrays.

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transformed Y or list of transformed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return [func(YThis) for YThis in Y]
    else:
        return func(Y)


def transposeIf(Y):
    """Transposes Y itself if Y is an array or each element of Y if it is a list/tuple of arrays.

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transposed Y or list of transposed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return [transposeIf(YThis) for YThis in Y]
    else:
        return Y.T


def subtractIf(X, Y):
    """Subtracts Y from X if X is an array, or subtracts each element of Y from
    each corresponding element of X if they are list/tuple of arrays.

    Args:
        X (np.array or list or tuple): input data or list of input data arrays.
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: X - Y or list of X - Ys
    """
    if Y is None:
        return X
    if isinstance(X, (list, tuple)):
        return [X[i] - Y[i] for i in range(len(X))]
    else:
        return X - Y


def catIf(Y, axis=None):
    """If Y is a list of arrays, will concatenate them otherwise returns Y

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transposed Y or list of transposed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return np.concatenate(Y, axis=axis)
    else:
        return Y


def prepare_fold_inds(num_folds, N):
    folds = []
    N_test = int(N / num_folds)
    for fold_ind in range(num_folds):
        test_inds = np.arange(
            N_test * fold_ind,
            N_test * (1 + fold_ind) if fold_ind < (num_folds - 1) else N,
        )
        train_inds = np.where(~np.isin(np.arange(N), test_inds))[0]
        folds.append(
            {
                "num_folds": num_folds,
                "fold": fold_ind + 1,
                "test_inds": test_inds,
                "train_inds": train_inds,
            }
        )
    return folds


def genCVFoldInds(N, CVFolds, shuffle=False, shuffle_seed=None):
    CVFoldInds = []
    allInds = np.arange(0, N)
    nTest = int(np.ceil(N / CVFolds))
    for fi in range(CVFolds):
        if fi == 0:
            n0 = 0
        else:
            n0 = CVFoldInds[fi - 1]["testInds"][-1] + 1
        testInds = allInds[np.logical_and(allInds >= n0, allInds < (n0 + nTest))]
        trainInds = allInds[
            np.logical_or(allInds < testInds[0], allInds > testInds[-1])
        ]
        CVFoldInds.append({"trainInds": trainInds, "testInds": testInds})
    if shuffle:
        allIndsPerm = np.arange(0, N)
        if shuffle_seed is not None:
            state = np.random.get_state()  # Back up global random seed
            np.random.seed(shuffle_seed)
        np.random.shuffle(allIndsPerm)
        if shuffle_seed is not None:
            np.random.set_state(state)  # Restore global random seed
        for fi in range(CVFolds):
            CVFoldInds[fi]["trainInds"] = allIndsPerm[CVFoldInds[fi]["trainInds"]]
            CVFoldInds[fi]["testInds"] = allIndsPerm[CVFoldInds[fi]["testInds"]]
    return CVFoldInds


def learnScaling(
    YTrain, removeMean, zScore, missing_marker=None, zscore_per_dim=True, axis=0
):
    """Learns a mean removal with or without zscore scaling

    Args:
        YTrain (np.array): data (time x dims)
        removeMean (bool): If True, will remove the mean even if zscore is False.
        zScore (bool): if True, will learn the scales required to zscore the data.
        zscore_per_dim (bool): if True, scale each dim with its own std, otherwise will scale all dims with the same scalar (std of flattened data)
        axis (int): axis over which to take mean/std
        missing_marker (np.floating): value to consider as missing. Default to None.

    Returns:
        _type_: _description_
    """
    if isinstance(YTrain, list):  # Trial based data provide as a list of numpy arrays
        # first concatenate the time dim
        YTrain = np.concatenate(YTrain, axis=axis)
    dim_axis = 1 if (axis == 0) else 0
    ny = YTrain.shape[dim_axis]
    yMean = np.zeros(ny)
    yStd = np.ones(ny)
    if removeMean or zScore:
        YTrainIsMissing = np.isnan(YTrain)
        if missing_marker is not None:
            YTrainIsMissing = np.logical_or(YTrainIsMissing, YTrain == missing_marker)
        YTrainMA = np.ma.array(YTrain, mask=YTrainIsMissing)
        yMean = np.array(np.ma.mean(YTrainMA, axis=axis))
        if zScore:
            yStd = np.array(np.ma.std(YTrainMA, axis=axis, ddof=1))
            if not zscore_per_dim:
                yStd = np.array(np.mean(yStd)) * np.ones_like(yMean)
    return yMean, yStd


def applyGivenScaling(X=None, mean=None, std=None, missing_marker=None, axis=0):
    if X is not None and (mean is not None or std is not None):
        if isinstance(X, (list, tuple)):
            X = [
                applyGivenScaling(thisX, mean, std, missing_marker, axis) for thisX in X
            ]
        else:
            X = np.copy(X)  # Do not modify input
            XIsMissing = np.isnan(X)
            if missing_marker is not None:
                XIsMissing = np.logical_or(XIsMissing, X == missing_marker)
            dim_axis = 1 if (axis == 0) else 0
            XIsOk = np.all(~XIsMissing, axis=dim_axis)
            if mean is not None:
                if axis == 0:
                    X[XIsOk, :] = X[XIsOk, :] - mean
                else:
                    X[:, XIsOk] = (X[:, XIsOk].T - mean).T
            if std is not None:
                if std.size == 1:
                    std = std * np.ones_like(mean)
                if axis == 0:
                    X[np.ix_(XIsOk, std > 0)] = X[np.ix_(XIsOk, std > 0)] / std[std > 0]
                else:
                    X[np.ix_(std > 0, XIsOk)] = (
                        X[np.ix_(std > 0, XIsOk)].T / std[std > 0]
                    ).T
    return X


def applyScaling(sId, X=None, meanField="xMean", stdField="xStd", missing_marker=None):
    if hasattr(sId, meanField):
        xMean = getattr(sId, meanField)
    else:
        xMean = None
    if hasattr(sId, stdField):
        xStd = getattr(sId, stdField)
    else:
        xStd = None
    return applyGivenScaling(X, xMean, xStd, missing_marker)


def undoScaling(
    sId, X=None, meanField="xMean", stdField="xStd", axis=0, missing_marker=None
):
    if X is not None and (hasattr(sId, meanField) or hasattr(sId, stdField)):
        if isinstance(X, (list, tuple)):
            X = [
                undoScaling(sId, thisX, meanField, stdField, axis, missing_marker)
                for thisX in X
            ]
        else:
            X = np.copy(X)  # Do not modify input
            XIsMissing = np.isnan(X)
            if missing_marker is not None:
                XIsMissing = np.logical_or(XIsMissing, X == missing_marker)
            dim_axis = 1 if (axis == 0) else 0
            XIsOk = np.all(~XIsMissing, axis=dim_axis)
            if hasattr(sId, stdField):
                std = getattr(sId, stdField)
                if std.size == 1:
                    std = std * np.ones((X.shape[dim_axis],))
                if axis == 0:
                    X[XIsOk, :] = X[XIsOk, :] * std
                else:
                    X[:, XIsOk] = X[:, XIsOk] * std[:, np.newaxis]
            if hasattr(sId, meanField):
                mean = getattr(sId, meanField)
                if axis == 0:
                    X[XIsOk, :] = X[XIsOk, :] + mean
                else:
                    X[:, XIsOk] = X[:, XIsOk] + mean[:, np.newaxis]
    return X


def sliceIf(Y, indexes):
    """Keep some indices from the first dim of np.array or from the items of a list

    Args:
        Y (np.array): returns a slice of an array
        indexes (list or np.array): indices to keep

    Returns:
        _type_: _description_
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return [Y[index] for index in indexes]
    else:
        return Y[indexes, ...]


def transposeIf(Y):
    """Transposes Y itself if Y is an array or each element of Y if it is a list/tuple of arrays.

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transposed Y or list of transposed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return [transposeIf(YThis) for YThis in Y]
    else:
        return Y.T


def discardSamples(Z, missing_marker, discardRatio=0, discardSeed=None):
    """Replaces some samples with missing_maker

    Args:
        Z (np.array): Data to discard samples from (time x dim)
        missing_marker (np.float): missing values will be indicated with this value
        discardRatio (float, optional): Ratio of samples to discard. Defaults to 0.
        discardSeed (float, optional): Seed for selecting samples to discard. Defaults to None.
            If None will keep equially distanced samples. Otherwise it will randomize the samples
            that are kept with the given seed.

    Returns:
        Z (np.array): The new data where samples are discarded at a discardRatio rate and replaced with missing_marker
    """
    if discardRatio > 0:
        ZD = np.empty(Z.shape)
        ZD[:] = missing_marker
        if discardSeed is not None:
            state = np.random.get_state()  # Back up global random seed
            np.random.seed(discardSeed)  # Set the seed for next line
            keepInds = np.where(np.random.rand(ZD.shape[0]) >= discardRatio)[
                0
            ]  # Keep random samples
            np.random.set_state(state)  # Restore global random seed
        else:
            missCount = int(np.round(1 / (1 - discardRatio)))
            keepInds = np.arange(
                missCount - 1, ZD.shape[0], missCount
            )  # Keep one sample after every missCount
        ZD[keepInds, :] = Z[keepInds, :]
        Z = ZD
    return Z


def get_trials_from_cat_data(dataIn, trial_samples=None):
    """Returns trials (trial x time x neuron) from concatenated data

    Args:
        data (np.array): concatenated data with time x neuron

    Returns:
        dataTrials (np.array): trial x time x neuron
    """
    data = copy.deepcopy(dataIn)
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    if (
        data is not None and len(data.shape) < 3
    ):  # Data is not in the 3d trial separated format
        if data.size > 0:
            # Assume first dimension is time, with trials concatenated with each other back to back
            dataTrials = data.reshape(
                (trial_samples, -1, data.shape[1]), order="F"
            ).transpose(1, 0, 2)
        else:
            dataTrials = np.zeros(
                (int(data.shape[0] / trial_samples), trial_samples, data.shape[1])
            )
    else:
        dataTrials = data
    return dataTrials


def get_cat_data_from_trials(dataTrialsIn, time_first=True):
    """Returns concatenated data from trial x time x neuron

    Args:
        dataTrials (np.array): trial x time x neuron
        dataTrials (np.array): if False, will assume that each element of the array has neuron x time dimensions, or the total dims to be trial x neuron x time

    Returns:
        catData (np.array): concatenated data with time x neuron
    """
    dataTrials = copy.deepcopy(dataTrialsIn)
    if isinstance(dataTrials, (list, tuple)):
        dataTrials = np.array(dataTrials)
    if dataTrials is not None and len(dataTrials.shape) == 3:
        if time_first:
            catData = dataTrials.transpose(1, 0, 2).reshape(
                (dataTrials.shape[0] * dataTrials.shape[1], dataTrials.shape[-1]),
                order="F",
            )
        else:
            catData = (
                dataTrials.transpose(2, 0, 1)
                .reshape(
                    (dataTrials.shape[0] * dataTrials.shape[-1], dataTrials.shape[1]),
                    order="F",
                )
                .T
            )
    else:
        catData = dataTrials
    return catData


def pickColumnOp(n, cols):
    """Returns a matrix multiplication operator that sets all but some columns of a matrix to zero
    when it is right-multiplied by the matrix.

    Args:
        n (int): number of columns
        col (list of int): list of column indices to keep

    Returns:
        [type]: [description]
    """
    M = np.zeros((n, n))
    M[cols, cols] = 1
    return M


def get_one_hot(targets, nb_classes):  # https://stackoverflow.com/a/42874726/2275605
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def powerset(iterable):  # https://docs.python.org/3/library/itertools.html
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def getIsOk(X=None, missing_marker=None):
    if X is not None:
        if missing_marker is not None:
            isOkX = np.logical_and(~np.isnan(X), X != missing_marker)
        else:
            isOkX = ~np.isnan(X)
    else:
        isOkX = None
    return isOkX


def isFlat(val, axis=0, thr=0):
    """Returns a bool array that is True for dimensions of axis=0 val that are flat

    Args:
        val (np.array): data
        axis (float, optional): axis over which flatness is evaluated. Defaults to 0.
        thr (float, optional): Threshold for determining flatness. Defaults to 0.

    Returns:
        np.array of bool: array of result for whether each dimension was flat or not
    """
    return (np.max(val, axis=axis) - np.min(val, axis=axis)) <= thr


def getInPeriodSamples(T, period):
    return np.nonzero(np.logical_and(T >= period[0], T < period[-1]))[0]


def extractEpochs(
    X,
    T,
    epochs,
    stretch=False,
    stretch_samples=None,
    min_samples=2,
    move_start_to_origin=False,
):
    allEData = []
    for ei, epoch in enumerate(epochs):
        eSamples = getInPeriodSamples(T, epoch["period"])
        if len(eSamples) >= min_samples:
            eData = X[eSamples, ...]
            if move_start_to_origin:
                eData = eData - eData[0, :]
            eTime = T[eSamples]
            eRelTime = eTime - eTime[0]
            allEData.append(
                {"data": eData, "time": eTime, "relTime": eRelTime, "epoch": epoch}
            )
    if stretch:
        if stretch_samples is None:
            stretch_samples = int(np.median([len(ed["time"]) for ed in allEData]))
        for ei, thisEData in enumerate(allEData):
            thisEData["data"], thisEData["time"] = resample(
                thisEData["data"], stretch_samples, t=thisEData["time"], axis=0
            )
            thisEData["relTime"] = thisEData["time"] - thisEData["time"][0]

    nSamples = np.array([len(ed["time"]) for ed in allEData])
    maxInd = np.argmax(nSamples)
    relTime = allEData[maxInd]["time"] - allEData[maxInd]["time"][0]

    D = np.empty((nSamples[maxInd], X.shape[1], len(nSamples)))
    D[:] = np.nan
    for ei, ed in enumerate(allEData):
        thisData = ed["data"]
        D[: thisData.shape[0], :, ei] = thisData

    ERP = np.nanmean(D, axis=-1)
    ERPStd = np.nanstd(D, axis=-1)
    ERPN = np.sum(~np.isnan(D), axis=-1)
    ERPSEM = ERPStd / np.sqrt(ERPN)

    return ERP, ERPSEM, D, relTime, allEData


def extractERP(
    X,
    T,
    event_times=None,
    epochs=None,
    epoch_timelock_ref="start",  # can also be 'end' or a number between 0 and 1
    rel_period=None,  # 2-element vector specifying the ERP extraction range relative to the event
    rel_samples=None,  # 2-element vector specifying the ERP extraction sample range relative to closest sample to the event
):
    if len(X.shape) == 1:
        X = np.atleast_2d(X).T
    if event_times is None:
        if (
            isinstance(epoch_timelock_ref, (int, float, np.ndarray))
            and epoch_timelock_ref >= 0
            and epoch_timelock_ref <= 1
        ):
            event_times = [
                e["period"][0] + epoch_timelock_ref * np.diff(e["period"])
                for e in epochs
            ]
        elif epoch_timelock_ref == "start":
            event_times = [e["period"][0] for e in epochs]
        elif epoch_timelock_ref == "end":
            event_times = [e["period"][-1] for e in epochs]
        else:
            raise (Exception("Unexpected option"))
    if rel_samples is not None:
        eRefSample = [np.argmin(np.abs(T - et)) for et in event_times]
        eSamples = [
            np.arange(rs + int(rel_samples[0]), rs + int(rel_samples[-1]))
            for rs in eRefSample
        ]
    else:
        if rel_period is None:
            eSamples = [getInPeriodSamples(T, e["period"]) for e in epochs]
        else:
            eSamples = [getInPeriodSamples(T, et + rel_period) for et in event_times]
    eSamples = [e[np.logical_and(e >= 0, e < len(T))] for e in eSamples]
    okInd = np.nonzero([e.size > 0 for e in eSamples])[0]
    eSamples = [eSamples[i] for i in okInd]
    event_times = [event_times[i] for i in okInd]
    eRefRelSample = [
        np.argmin(np.abs(T[eSamples[ei]] - event_times[ei]))
        for ei in range(len(event_times))
    ]
    eSamplesCnt = [ind.size for ind in eSamples]
    longestEInd = np.argmax(eSamplesCnt)
    relTime = (
        T[eSamples[longestEInd]] - T[eSamples[longestEInd][eRefRelSample[longestEInd]]]
    )
    D = np.empty((np.max(eSamplesCnt), X.shape[1], len(event_times)))
    D[:] = np.nan
    for ei in range(len(event_times)):
        if (
            isinstance(epoch_timelock_ref, (int, float, np.ndarray))
            and epoch_timelock_ref >= 0
            and epoch_timelock_ref <= 1
        ):
            rel_times = (
                T[eSamples[ei]] - event_times[ei]
            )  # Times of this epoch, relative to the epoch_timelock_ref
            st_ind = np.argmax(
                np.convolve(relTime, np.flip(rel_times), mode="valid")
            )  # Best alignment between relative
            # times of this epoch with the ERP
            D[st_ind : (st_ind + eSamplesCnt[ei]), ..., ei] = X[eSamples[ei], ...]
        elif epoch_timelock_ref == "start":
            D[: eSamplesCnt[ei], ..., ei] = X[eSamples[ei], ...]
        elif epoch_timelock_ref == "end":
            D[-eSamplesCnt[ei] :, ..., ei] = X[eSamples[ei], ...]
        else:
            raise (Exception("Unexpected option"))

    ERP = np.nanmean(D, axis=-1)
    ERPStd = np.nanstd(D, axis=-1)
    ERPN = np.sum(~np.isnan(D), axis=-1)
    ERPSEM = ERPStd / np.sqrt(ERPN)
    return ERP, ERPSEM, D, relTime


def standardizeStateTrajectory(
    xTest,
    tTest=None,
    alignEpochs=None,
    remove_mean=True,
    generate_plot=False,
    rotate_end_point=False,
):
    """Finds a linear transformation that standardizes a 2D or 3D trajectory. Useful for aggregating info
    across folds/sessions for models with latent states. Before aggregating, you need to make the latent
    states similar, which this function aims to achieve by standarding them. 4 steps are performed by
    default:
    1) All states dimensions are z-scored to have mean of zero and unit variance.
    2) [For 3D states], a rotation is perform to bring most of the trial average trajectory into the xy plane.
    3) A rotation is performed to bring the starting point of the trial average trajectory on the x-axis.
    4) If the trajectory is largely clockwise (when projected onto xy-plane), the y-axis is mirrored.

    Args:
        xTest (numpy array): time x dimensions. Can have 2 or 3 columns.
        tTest (numpy array): time x 1. The times associated by the above samples. Used to extract trials.
                Needed if there are multiple epochs in the data. Defaults to None.
        alignEpochs (list of epochs): will be passed to extractEpochs to extract trial average across
                these epochs. That trial average will be the basis of standardization. If None will treat
                the data as one single epoch/trial. Defaults to None.
        remove_mean (bool): If False, will not remove the mean of the data. Defaults to True.
        generate_plot (bool): If True, will generate some plots to show the steps. Defaults to False.
        rotate_end_point (bool): If True, will rotate end point of trajectory to land on 0 degrees, otherwise will rotate the starting point.

    Returns:
        E (numpy array): The matrix that describes the final similarity transform. x_new = E * x
        xTestN (numpy array): The transformed version of the states. xTestN = (E @ (xTest-X0).T).T
    """
    nx = xTest.shape[-1]
    if nx not in [1, 2, 3]:
        raise (Exception("Making states similar is only supported for nx=1, 2, or 3"))

    if remove_mean:
        X0 = np.mean(xTest, axis=0)
    else:
        X0 = np.zeros(nx)

    # Construct the transform to be applied to x
    # Scale all dims to unit variance
    xStd = np.std(xTest, axis=0)
    E1 = np.diag(1 / xStd)

    xZScored = (E1 @ (xTest - X0).T).T

    if alignEpochs is not None:
        xMean = extractEpochs(
            xZScored, tTest, epochs=alignEpochs, stretch=True, min_samples=4
        )[0]
    else:
        xMean = xZScored
    xMeanBU = np.copy(xMean)

    x0 = xMean[0, :] if not rotate_end_point else xMean[-1, :]
    if nx == 3:
        # Rotate around y-axis to make plane normal to z-axis
        planeNormal1, xMean1Centeroid = find_best_plane_fit(xMean)
        R1 = rotation_matrix_from_vectors_3d(
            planeNormal1, np.array([0, 0, np.linalg.norm(planeNormal1)])
        )
        xMean1 = (R1 @ xMean.T).T

        # Rotate to put the start of reach epochs at on the x axis
        x0 = xMean1[0, :]
        # R2 = rotation_matrix_from_vectors_3d(x0, np.array([np.linalg.norm(x0[:2]),0,x0[-1]]))
        # Rotate to put the start of reach epochs at (1, 0)
        theta = -np.angle(x0[0] + 1j * x0[1])
        R2_XY = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        R2 = np.array(
            [[R2_XY[0, 0], R2_XY[0, 1], 0], [R2_XY[1, 0], R2_XY[1, 1], 0], [0, 0, 1]]
        )
        # R2 = np.eye(3)
        xMean2 = (R2 @ xMean1.T).T

        R = R2 @ R1

        # Mirror the y axis to make counter-clockwise when projected onto the xy axis
        isCWXY = isClockwise(xMean2[np.ix_(np.arange(xMean2.shape[0]), [0, 1])])
        if isCWXY:
            E3_XY = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            E3_XY = np.eye(3)
        # E3_XY = np.eye(3)

        E3 = E3_XY
    elif nx == 2:
        # Rotate to put the start of reach epochs at (1, 0)
        theta = -np.angle(x0[0] + 1j * x0[1])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Flip dim 2 if needed to make rotation counter-clockwise (CC)
        xMean1 = (R @ xMean.T).T
        isCW = isClockwise(xMean1)
        if isCW:
            E3 = np.array([[1, 0], [0, -1]])
        else:
            E3 = np.eye(2)
    elif nx == 1:
        R = np.array([[+1]])  # No rotation needed

        # Flip dim 1 if needed to make the start point start from x > 0
        if xMean[0] < 0:
            E3 = np.array([[-1]])
        else:
            E3 = np.array([[+1]])
    else:
        raise (Exception(f"nx={nx} is not supported"))

    # Get the overall similarity transform
    E = E3 @ R @ E1

    if generate_plot:
        xMeanF = (E3 @ R @ xMeanBU.T).T
        if nx == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xMeanBU[:, 0], color="b", label="original")
            ax.scatter(0, xMeanBU[0, 0], color="b")
            ax.plot(xMeanF[:, 0], color="k", label="after all steps")
            ax.scatter(0, xMeanF[0, 0], color="k")
            ax.set_xlabel("Sample")
            ax.set_ylabel("value")
            ax.legend()
            plt.show()
        elif nx == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xMeanBU[:, 0], xMeanBU[:, 1], color="b", label="original")
            ax.scatter(xMeanBU[0, 0], xMeanBU[0, 1], color="b")
            ax.plot(xMean1[:, 0], xMean1[:, 1], color="r", label="after step 1")
            ax.scatter(xMean1[0, 0], xMean1[0, 1], color="r")
            ax.plot(xMeanF[:, 0], xMeanF[:, 1], color="k", label="after all steps")
            ax.scatter(xMeanF[0, 0], xMeanF[0, 1], color="k")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif nx == 3:
            raise NotImplementedError

    # Apply the transform on the states and return the updated states
    xTestN = (E @ (xTest - X0).T).T
    return xTestN, E, X0


def getGapInds(time, max_gap=None):
    """
    Finds gaps in time series data (e.g. time) that are larger than a specified magnitude
    Outputs:
    - (1) preGapInds: indices immediately before the gaps
    - (2) insertInds: indices to be used with the np.insert command to fill in the gaps
    """
    tDiff = np.diff(time, axis=0)
    if max_gap is None:
        max_gap = np.median(tDiff, axis=0) * 2
    preGapInds = np.nonzero(tDiff > max_gap)[0]
    return preGapInds


def find_best_plane_fit(data):
    """finds the normal to the best hyperplane fit to data

    Args:
        data ([type]): samples x dimension (100 x 3)

    Returns:
        normal vector (numpy array): the normal vector of the plane
        centeroid (numpy array): the normal of the plane
    """
    # Fit a plane to the data to get it's normal vector:
    # Based on https://math.stackexchange.com/a/99317/115169
    centeroid = np.mean(data, axis=0)
    xDeMean = data - centeroid
    svdU = np.linalg.svd(xDeMean.T)[0]
    planeNormal = svdU[:, -1]
    return planeNormal, centeroid


def getPlaneZFunc(normal, point):
    d = -point @ normal
    planeZFunc = lambda x, y: (-d - normal[0] * x - normal[1] * y) / normal[-1]
    return planeZFunc


# https://stackoverflow.com/a/59204638/2275605
def rotation_matrix_from_vectors_3d(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def isClockwise(data):
    """Returns true if a 2d trajectory appears to be largely counter-clockwise

    Args:
        data (numpy array): time x dimensions (e.g. 100 x 2)

    Returns:
        isCW (bool): True if counter clockwise
    """
    XY1 = data[0, :]
    XY2 = data[-1, :]
    lineFunc = lambda x: XY1[1] + (x - XY1[0]) * (XY2[1] - XY1[1]) / (XY2[0] - XY1[0])
    xCloser = np.linspace(XY2[0], XY1[0], 10)
    xCat = np.concatenate((data[:, 0], xCloser), axis=0)
    yCat = np.concatenate((data[:, 1], lineFunc(xCloser)), axis=0)
    xCatDiff = np.diff(xCat)
    yCatMid = (yCat[:-1] + yCat[1:]) / 2
    integral = xCatDiff * (yCatMid - np.min(yCatMid))
    isCW = np.nanmean(integral) > 0
    # Incorrect old approach for reference:
    # deltaYFromLine = data[:, 1] - lineFunc( data[:, 0] )
    # isCW = np.nanmean(deltaYFromLine) < 0
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(xCat, yCat, '--')
    # plt.plot(data[:, 0], lineFunc(data[:, 0]))
    plt.scatter(data[0, 0], data[0, 1])
    plt.title('Is clock-wise? {}'.format(isCW))
    plt.show()
    #"""
    return isCW


class TimeRemappingRule:
    def __init__(self, period=None, func=None):
        self.period = period
        self.func = func

    def doesItApply(self, t):
        if self.period is None:
            return True
        if isinstance(self.period, (tuple, list, np.ndarray)):
            return np.logical_and(t >= self.period[0], t < self.period[-1])
        raise (Exception("Unexpected period!"))

    def apply(self, t):
        if isinstance(self.func, type(np.mean)):
            t = self.func(t)
        return t


class TimeRemapper:
    def __init__(self):
        self.rules = []

    def apply(self, t):
        tNew = np.array(t)
        for rule in self.rules:
            applies = rule.doesItApply(t)
            tNew[applies] = rule.apply(tNew[applies])
        return tNew


def shortenGaps(time, max_gap=None, shortened_value=None):
    tDiff = np.diff(time, axis=0)
    if max_gap is None:
        max_gap = np.median(tDiff, axis=0) * 4
    if shortened_value is None:
        shortened_value = max_gap
    preGapInds = getGapInds(time, max_gap)
    timeRemapper = TimeRemapper()
    timeBU = np.array(time)
    for i in preGapInds:
        dt = -time[i + 1] + time[i] + shortened_value
        time[(i + 1) :] = time[(i + 1) :] + dt
        timeRemapper.rules.append(
            TimeRemappingRule(
                period=(np.mean(timeBU[i : (i + 2)]), np.inf),
                func=lambda x, dt_in=dt: np.array(x + dt_in),
            )
        )
    # timeBURedo = timeRemapper.apply(timeBU)
    # np.testing.assert_allclose(time, timeBU)
    return time, timeRemapper


def autoDetectSignalType(Z):
    if Z is None:
        return None
    if "float" in str(Z.dtype):
        ZType = "cont"
    elif "int" in str(Z.dtype):
        ZType = "cat"
    else:
        raise (Exception("Not supported"))
    return ZType

def prepAxesStyle(ax, grid_on=True, box_on=True):
    """Prepares the axes style by adding grids and removing spines

    Args:
        ax (_type_): _description_
        grid_on (bool, optional): _description_. Defaults to True.
        box_on (bool, optional): _description_. Defaults to True.
    """    
    if grid_on:
        ax.grid(alpha=0.1)
    if box_on:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in', length=2)
