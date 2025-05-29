""" 
Copyright (c) 2022 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

""" Implements DPAD training with flexible nonlinearity  """
""" Runs a search over potential location/types for nonliearity and picks the
    best via an inner cross-validation within the training data """

import copy
import itertools
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import sys
import time
from collections import ChainMap
from datetime import datetime
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import xxhash
from PSID import LSSM
from PSID import MatHelper as mh

from .. import DPADModel
from .evaluation import evalSysId, evaluateDecoding, getCVedPerf, sanitize_perfs
from .file_tools import bytes_to_string, pickle_load, pickle_save
from .parse_tools import (
    extractFloatsFromRegex,
    extractIntsFromRegex,
    extractLinearRangesFromRegex,
    extractNumberFromRegex,
    extractPowRangesFromRegex,
    extractStrsFromRegex,
    extractValueRanges,
    parseInnerCVFoldSettings,
    parseMethodCodeArg_kpp,
    parseMethodCodeArgEnsemble,
    parseMethodCodeArgStepsAhead,
)
from .tools import (
    applyGivenScaling,
    discardSamples,
    genCVFoldInds,
    learnScaling,
    powerset,
    sliceIf,
    transposeIf,
    undoScaling,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import coloredlogs

    logFmt = "%(asctime)s [%(levelname)s] [%(filename)s > %(lineno)s] - %(message)s"
    coloredlogs.install(
        level=logging.INFO,
        fmt=logFmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        logger=logger,
    )
    print(f"Logging handlers: {logger.parent.handlers}")


def get_data_hash(*args) -> list[str]:
    """Returns a list of hashes for the data

    Returns:
        data_hashes (list[str]): _description_
    """
    data_hashes = []
    for data in args:
        if data is not None:
            data_hashes.append(xxhash.xxh32(data).hexdigest())
        else:
            data_hashes.append("N")
    return data_hashes


def fitDPADWithFlexibleNonlinearity(
    Y,
    Z=None,
    U=None,
    nx=None,
    n1=None,
    methodCode="DPAD_GSUT_iCVF2_LSTM_uAKCzCy3HL128U",  # Search for
    settings=None,
    saveDir=os.path.join(".", "results"),
    saveFName=None,
    addDataHashToSavePath=True,  # adds data hash to save dir to make sure we don't load the model for a different data down the road
    addMethodCodeToSavePath=True,  # adds data hash to save dir to make sure we don't load the model for a different data down the road
):
    """_summary_

    Args:
        Y (_type_): _description_
        Z (_type_, optional): _description_. Defaults to None.
        U (_type_, optional): _description_. Defaults to None.
        nx (_type_, optional): _description_. Defaults to None.
        n1 (_type_, optional): _description_. Defaults to None.
        methodCode (str, optional): _description_. Defaults to 'DPAD_GSUT_iCVF2_uAKCzCy1HL64U'.
        saveDir (_type_, optional): _description_. Defaults to os.path.join('.', 'results').
        saveFName (str, optional): _description_. Defaults to ''.
        addDataHashToSavePath (str, optional): if True, will add a data hash to the save path. Defaults to True.
        addMethodCodeToSavePath (str, optional): if True, will add methodCode as a directory in the save path. Defaults to True.

    Returns:
        _type_: _description_
    """
    if settings is None:
        settings = {}

    if addDataHashToSavePath:
        data_hashes = get_data_hash(Y, Z, U)
        saveDir = os.path.join(saveDir, "_".join(data_hashes))

    if addMethodCodeToSavePath:
        saveDir = os.path.join(saveDir, methodCode)

    if saveFName is None:
        saveFName = "" if addMethodCodeToSavePath else methodCode

    if not saveFName.endswith(".p"):
        saveFName += ".p"

    if "iCVF" in methodCode:
        regex = r"_iCVF(\d*)"  # iCVF0
        matches = re.finditer(regex, methodCode)
        for matchNum, match in enumerate(matches, start=1):
            icvf = match.groups()
        iCVFolds = int(icvf[0])
        settings["iCVFolds"] = iCVFolds

    subMethods, methodCodeBase = prepareHyperParameterSearchSpaceFromMethodCode(
        methodCode
    )
    print(
        f"Considering {len(subMethods)} possible combinations of location/type for nonlinearities (this might take a while)..."
    )
    iCVRes = pickMethodHyperParamsWithInnerCV(
        Y,
        ZTrain=Z,
        UTrain=U,
        methodCode=methodCode,
        subMethods=subMethods,
        nx=nx,
        n1=n1,
        settings=settings,
        WS={},
        saveFilePath=os.path.join(saveDir, saveFName),
    )
    selectedMethodCode = iCVRes["selectedMethodCode"]  # Best method code
    return selectedMethodCode, iCVRes


def pickMethodHyperParamsWithInnerCV(
    YTrain,
    ZTrain=None,
    UTrain=None,
    TTrain=None,
    methodCode=None,
    subMethods: None | list[str] = None,
    nx=None,
    n1=None,
    settings={},
    WS={},
    subMethodArgs=None,
    subMethodSettings=None,
    YType=None,
    ZType=None,
    ZClasses=None,
    missing_marker=None,
    criteria="within_sem",
    peak_det_sem_multiplier=1,
    peak_det_ratio=0.05,
    saveFilePath="",
):
    """Fits models with different methodCodes (as specified by subMethod), performs inner
    cross-validation for each, and picks one based on neural-behavioral prediction accuracy

    Args:
        YTrain (np.ndarray): signal 1 (e.g. neural activity)
        ZTrain (np.ndarray, optional): signal 2 (e.g. behavior). Defaults to None.
        UTrain (np.ndarray, optional): input (e.g. task instructions). Defaults to None.
        TTrain (np.ndarray, optional): time vector. size must be consistent with Y,Z,U. Defaults to None.
        methodCode (string, optional): method code. Defaults to None.
        subMethods (list of string, optional): list of method codes to consider.
            If None, will just consider methodCode and skip this function. Defaults to None.
        nx (list of int, optional): state dimension nx values to fit. Defaults to None.
        n1 (list of int, optional): state dimension in stage 1 (n1) value(s) to use. Defaults to None.
        settings (dict, optional): model fitting settings. Defaults to {}.
        WS (dict, optional): _description_. Defaults to {}.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        ZClasses (np.array, optional): class values in Z. Defaults to None.
        missing_marker (number, optional): the marker value for missing data. Defaults to None.
        saveFilePath (string, optional): path to save file. Defaults to ''.

    Returns:
        iCVRes (dict): inner cross-validation results
    """
    prioritizeZPred = "GSUTy" not in methodCode  # and 'GSUT' in methodCode and
    zDefaultPerfMeasure = "meanAUC" if ZType == "cat" else "CC" #"meanCC"
    yDefaultPerfMeasure = "yCC" #"meanyCC"
    if "iCVSelPerfMeasure" not in settings:
        settings["iCVSelPerfMeasure"] = (
            zDefaultPerfMeasure if prioritizeZPred else yDefaultPerfMeasure
        )
    if "iCVSelPerfMeasure2" not in settings:
        settings["iCVSelPerfMeasure2"] = (
            yDefaultPerfMeasure if prioritizeZPred else zDefaultPerfMeasure
        )  # Among those with 1 sem of iCVSelPerfMeasure, pick one that is within 1sem of the best iCVSelPerfMeasure2
    if "iCVSelIn1SEM" not in settings:
        settings["iCVSelIn1SEM"] = (
            True  # If true, will select the smallest nx (or the best iCVSelPerfMeasure2) that reaches within 1 sem of the best innerCV iCVSelPerfMeasure
        )
    if "iCVSelIn1SEM2" not in settings:
        settings["iCVSelIn1SEM2"] = (
            False  # If true, will select the smallest nx that reaches within 1 sem of the best innerCV iCVSelPerfMeasure2, among those that satisfy sem conditions for iCVSelPerfMeasure
        )
    if "iCVFolds" not in settings:
        CVFolds = settings["CVFolds"] if "CVFolds" in settings else 5
        settings["iCVFolds"] = np.max((2, CVFolds - 1))
    if (
        "iCVFoldsToConsider" not in settings
    ):  # The folds to run and consider in our final selection of hyperparameters
        settings["iCVFoldsToConsider"] = None

    iCVRes = {}
    if subMethods is None:
        subMethods = [methodCode]

    if len(subMethods) == 1:
        logger.info(
            f"Only one case for inner CV ({subMethods})... choice it already clear. Running normally..."
        )
        iCVRes = {"selectedMethodCode": subMethods[0]}
    else:
        logger.info(
            "Running innerCV to choose among the following {} variants:\n{}".format(
                len(subMethods),
                "\n".join(
                    "({}) {}".format(mi + 1, ms) for mi, ms in enumerate(subMethods)
                ),
            )
        )
        methodCodesICV = copy.copy(subMethods)
        if (
            settings["iCVSelPerfMeasure"] in ["meanCC", "meanR2", "meanNRMSE"]
            and settings["iCVSelPerfMeasure2"] is None
        ):
            methodCodesICV = [
                (m + "_skipCy") if "skipCy" not in m and ("DPAD" in m) else m
                for m in methodCodesICV
            ]
        settingsThis = copy.deepcopy(settings)
        settingsThis["ordersToSearch"] = [nx]
        settingsThis["CVFolds"] = (
            settings["iCVFolds"]
            if "iCVFolds" in settings and settings["iCVFolds"] >= 0
            else settings["CVFolds"] - 1
        )
        settingsThis["trainingDataPortionToKeep"] = None
        settingsThis["foldsToRun"] = (
            settings["iCVFoldsToRun"]
            if (
                "iCVFoldsToRun" in settings
                and settings["iCVFoldsToRun"] is not None
                and settings["iCVFoldsToRun"] != -1
            )
            else []
        )
        settingsThis["foldsToConsider"] = settings["iCVFoldsToConsider"]
        settingsThis["methodIndsToRun"] = (
            settings["iCVMethodIndsToRun"]
            if "iCVMethodIndsToRun" in settings
            and settings["iCVMethodIndsToRun"] is not None
            else []
        )
        settingsThis["genDataFigs"] = False
        settingsThis["removeFoldResults"] = False
        settingsThis["xTrajPlotNx"] = []
        settingsThis["findFixedPoints"] = False
        settingsThis["computePerfInTraining"] = False

        # subMethodArgs=None,
        # if subMethodSettings is not None:
        #     settingsThisGiven = subMethodSettings[]

        idSysAll, perfAll, CVFitRes = doCVedModelFit(
            YTrain,
            ZTrain,
            methodCodesICV,
            settingsThis["CVFolds"],
            nxVals=None,
            n1Vals=[n1],
            U=UTrain,
            trueSys=None,
            saveFile=saveFilePath,
            CVFoldInds=None,
            T=TTrain,
            YType=YType,
            ZType=ZType,
            settings=settingsThis,
        )

        if idSysAll is None:
            raise (Exception("Not all innerCV results are ready. Exiting..."))

        logger.info(
            "Selecting among the {} tried method variants based on the inner CV results: ".format(
                len(settingsThis["casesToRun"])
            )
        )
        perfAll = CVFitRes["perfAll"]  # Dims: method, fold, nx
        okPerfs = [
            pm
            for pm in list(perfAll.flatten()[0].keys())
            if pm == settings["iCVSelPerfMeasure"]
            or re.findall(
                r"^_(\d)+step$", pm.replace(settings["iCVSelPerfMeasure"], "")
            )
        ]
        if len(okPerfs) == 0:
            okPerfs = [
                pm
                for pm in list(perfAll.flatten()[0].keys())
                if pm == settings["iCVSelPerfMeasure2"]
                or re.findall(
                    r"^_(\d)+step$", pm.replace(settings["iCVSelPerfMeasure2"], "")
                )
            ]
        if settings["iCVSelPerfMeasure"] in okPerfs:
            iCVSelPerfMeasure = settings["iCVSelPerfMeasure"]
        elif settings["iCVSelPerfMeasure2"] in okPerfs:
            iCVSelPerfMeasure = settings["iCVSelPerfMeasure2"]
            logger.warning(
                settings["iCVSelPerfMeasure"]
                + " not available so using {iCVSelPerfMeasure} instead to pick hyperparameters"
            )
        else:
            iCVSelPerfMeasure = okPerfs[0]
            logger.warning(
                settings["iCVSelPerfMeasure"]
                + " not available so using {iCVSelPerfMeasure} instead to pick hyperparameters"
            )
        if 'mean' not in iCVSelPerfMeasure:
            yValsF = np.vectorize(lambda a: np.nanmean(a[iCVSelPerfMeasure]))(perfAll)
        else:
            yValsF = np.vectorize(lambda a: a[iCVSelPerfMeasure])(perfAll)
        if not settings["iCVSelIn1SEM"]:
            logger.info(
                "- Preaveraging {} across the {} folds => sem will be 0 and will pick the case with best {}".format(
                    settings["iCVSelPerfMeasure"],
                    yValsF.shape[1],
                    settings["iCVSelPerfMeasure"],
                )
            )
            yValsF = np.mean(yValsF, 1)[:, np.newaxis]
        if settings["iCVSelPerfMeasure2"] is not None:
            okPerfs = [
                pm
                for pm in list(perfAll.flatten()[0].keys())
                if pm == settings["iCVSelPerfMeasure2"]
                or re.findall(
                    r"^_(\d)+step$", pm.replace(settings["iCVSelPerfMeasure2"], "")
                )
            ]
            if settings["iCVSelPerfMeasure2"] in okPerfs:
                iCVSelPerfMeasure2 = settings["iCVSelPerfMeasure2"]
            elif len(okPerfs) > 0:
                iCVSelPerfMeasure2 = okPerfs[0]
                logger.warning(
                    settings["iCVSelPerfMeasure2"]
                    + " not available so using {iCVSelPerfMeasure2} instead to pick hyperparameters"
                )
            else:
                iCVSelPerfMeasure2 = None
            if iCVSelPerfMeasure2 is not None:
                if 'mean' not in iCVSelPerfMeasure2:
                    yValsF2 = np.vectorize(lambda a: np.nanmean(a[iCVSelPerfMeasure2]))(perfAll)
                else:
                    yValsF2 = np.vectorize(lambda a: a[iCVSelPerfMeasure2])(perfAll)
                if not settings["iCVSelIn1SEM2"]:
                    logger.info(
                        "- Preaveraging {} across the {} folds => sem will be 0 and will pick the case with best {}".format(
                            settings["iCVSelPerfMeasure2"],
                            yValsF2.shape[1],
                            settings["iCVSelPerfMeasure2"],
                        )
                    )
                    yValsF2 = np.mean(yValsF2, 1)[:, np.newaxis]
            else:
                yValsF2 = None
                iCVSelPerfMeasure2 = settings["iCVSelPerfMeasure2"]
        else:
            yValsF2 = None
            iCVSelPerfMeasure2 = settings["iCVSelPerfMeasure2"]
        yVals, yValsCVSEM, bestMethodInd, bestIndsAll = getCVedPerf(
            yValsF,
            iCVSelPerfMeasure,
            xValAxis=[0],
            preMeanAxes=[1],
            yValsF2=yValsF2,
            perfField2=iCVSelPerfMeasure2,
            verbose=True,
            peak_det_criteria=criteria,
            peak_det_sem_multiplier=peak_det_sem_multiplier,
            peak_det_ratio=peak_det_ratio,
        )

        subMethodsDone = settingsThis["casesToRun"]
        if "skipCy" not in methodCode:
            subMethodsDone = [mc.replace("_skipCy", "") for mc in subMethodsDone]
        logger.info(
            "Picked the following method after inner CV: (index {}) {}".format(
                bestMethodInd[0], subMethodsDone[bestMethodInd[0]]
            )
        )

        iCVRes = {
            "selectedMethodCode": subMethodsDone[bestMethodInd[0]],
            "settings": settingsThis,
            "CVFitRes": CVFitRes,
            "nxVals": [nx],
            "selectedInd": bestMethodInd,
            "bestIndsAll": bestIndsAll,
        }
    return iCVRes


def doCVedModelFit(
    Y,
    Z,
    methodCodes,
    CVFolds,
    nxVals,
    U=None,
    n1Vals=None,
    trueSys=None,
    saveFile=None,
    CVFoldInds=None,
    T=None,
    YType=None,
    ZType=None,
    UType=None,
    ZClasses=None,
    settings={},
    comparisonRef=None,
):
    """Performs cross-validated model fitting to neural data Y and behavior data Z

    Args:
        Y (np.array): input data.
        Z (np.array): output data.
        methodCodes (list of string): list of method codes.
        CVFolds (number): number of folds.
        nxVals (list of number): list of nx values.
        U (np.array, optional): external input. Defaults to None.
        n1Vals (list of number, optional): n1 values. Defaults to None.
        trueSys (LSSM, optional): true model, if known in simulations. Defaults to None.
        saveFile (string, optional): base path to the saved file. Defaults to None.
        CVFoldInds (list of dict): list of training/test sets for folds. Defaults to None.
        T (np.array, optional): time series. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        UType (string, optional): data type of U. Defaults to None.
        ZClasses (np.array): class values in Z. Defaults to None.
        settings (dict, optional): model fitting results. Defaults to {}.

    Returns:
        idSysAll (list of dict): learned models
        perfAll (list of dict): performance measures for learned models
        CVFitRes (dict): fit results
    """
    Y, Z, U, YType, ZType, UType, ZClasses = prep_data(
        Y, Z, U, YType, ZType, UType, settings
    )

    settings["casesToRun"] = methodCodes  # Backward compatibility
    if nxVals is None:
        nxVals = settings["ordersToSearch"]
    settings["ordersToSearch"] = nxVals  # Backward compatibility

    if "yDiscardRatio" not in settings:
        settings["yDiscardRatio"] = 0
    if "yDiscardSeed" not in settings:
        settings["yDiscardSeed"] = None  # None: non-random discarding
    if "zDiscardRatio" not in settings:
        settings["zDiscardRatio"] = 0
    if "zDiscardSeed" not in settings:
        settings["zDiscardSeed"] = None  # None: non-random discarding
    if "missing_marker" not in settings:
        settings["missing_marker"] = -1.0
    missing_marker = settings["missing_marker"]

    if "zScoreY" not in settings:
        settings["zScoreY"] = True
    if "zScoreZ" not in settings:
        settings["zScoreZ"] = True
    if "zScoreU" not in settings:
        settings["zScoreU"] = True
    if "removeYMean" not in settings:
        settings["removeYMean"] = True
    if "removeZMean" not in settings:
        settings["removeZMean"] = True
    if "removeUMean" not in settings:
        settings["removeUMean"] = True

    if "useXFilt" not in settings:
        settings["useXFilt"] = False

    if "PSIDKKF_missingZStrategy" not in settings:
        settings["PSIDKKF_missingZStrategy"] = (
            "interpolate"  # Can be 'interpolate' or 'discard'
        )

    if "horizon" not in settings:
        settings["horizon"] = [10]

    settings["CVFolds"] = CVFolds  # Backward compatibility
    if "foldsToRun" not in settings or (-1 in settings["foldsToRun"]):
        settings["foldsToRun"] = []  # If empty will run all

    if "foldsToConsider" not in settings:
        settings["foldsToConsider"] = (
            None  # If None will keep all folds. If a list, will exclude other folds after dividing data up ino folds
        )

    if "iCVFoldsToConsider" not in settings:
        settings["iCVFoldsToConsider"] = None
    if (
        settings["iCVFoldsToConsider"] is not None
        and len(settings["iCVFoldsToConsider"]) > 0
    ):
        if len(settings["foldsToRun"]) == 0:
            settings["foldsToRun"] = list(range(1, 1 + CVFolds))
        settings["foldsToRun"] = [
            foldNum
            for foldNum in list(range(1, 1 + CVFolds))
            if foldNum in settings["iCVFoldsToConsider"]
        ]

    if "methodIndsToRun" not in settings or (-1 in settings["methodIndsToRun"]):
        settings["methodIndsToRun"] = []  # If empty will run all

    if CVFoldInds is None:
        if CVFolds > 0:
            CVFoldInds = genCVFoldInds(
                len(Y) if isinstance(Y, list) else Y.shape[0], CVFolds
            )
        else:
            allInds = np.arange(0, Y.shape[0])
            CVFoldInds = [{"trainInds": allInds, "testInds": allInds}]
            CVFolds, settings["CVFolds"] = 1, 1

        if (
            "trainingDataPortionToKeep" in settings
            and settings["trainingDataPortionToKeep"] is not None
        ):
            if not isinstance(settings["trainingDataPortionToKeep"], (list, tuple)):
                settings["trainingDataPortionToKeep"] = (
                    0,
                    settings["trainingDataPortionToKeep"],
                )
            dataPortionToKeep = settings["trainingDataPortionToKeep"]
            for fi, foldInds in enumerate(CVFoldInds):
                isTrain = foldInds["trainInds"]
                keepSamples = np.arange(
                    int(np.floor(dataPortionToKeep[0] * isTrain.size)),
                    int(np.ceil(dataPortionToKeep[1] * (1 + isTrain.size))),
                )
                logger.warning(
                    f"For fold {fi+1}/{len(CVFoldInds)}, will keep only {len(keepSamples)}/{isTrain.size} ({len(keepSamples)/isTrain.size*100:.3g}%) training samples (i.e., samples {keepSamples[0]} to {keepSamples[-1]}) of the training data"
                )
                foldInds["trainInds"] = isTrain[keepSamples, ...]

    settingsBU = copy.deepcopy(settings)

    idSysAll, perfAll, CVFitRes = loadAndParseFullCVedResults(
        saveFile, Z, methodCodes, CVFolds, nxVals, CVFoldInds, settings, missing_marker
    )

    folds = np.arange(1, 1 + CVFolds)
    if len(settings["foldsToRun"]) < 1:
        settings["foldsToRun"] = np.copy(folds)
    foldsToRun = settings["foldsToRun"]
    methodIndsToRun = settings["methodIndsToRun"]

    if idSysAll is not None:
        addMultistepPerfs(
            Y,
            Z,
            methodCodes,
            nxVals,
            U,
            saveFile,
            CVFoldInds,
            YType,
            ZType,
            settings,
            missing_marker,
            idSysAll,
            CVFitRes,
            foldsToRun,
        )
        perfAll, CVFitRes = addTrainingPerformanceIfMissing(
            Y,
            Z,
            methodCodes,
            nxVals,
            U,
            T,
            saveFile,
            CVFoldInds,
            YType,
            ZType,
            settings,
            missing_marker,
            idSysAll,
            perfAll,
            CVFitRes,
            CVFolds,
            foldsToRun,
        )
    else:
        logger.info(
            "Assuming new run because no result file was found in {}".format(saveFile)
        )
        perfAll = None
        if "skipIfNotDone" in settings and settings["skipIfNotDone"]:
            return None, None, None

    if perfAll is None:
        idSysAll, perfAll, CVFitRes = doCVedModelingForGivenFolds(
            Y,
            Z,
            U,
            methodCodes,
            nxVals,
            n1Vals,
            trueSys,
            CVFolds,
            CVFoldInds,
            foldsToRun,
            methodIndsToRun,
            T,
            YType,
            ZType,
            UType,
            ZClasses,
            settings,
            missing_marker,
            saveFile,
        )

    # Extract inner CV stats
    if "idInfoAll" in CVFitRes:
        CVFitRes["idInfoAll"] = addSelectedMethodInfoToIdInfo(
            CVFitRes["idInfoAll"], CVFitRes["extraSettings"]
        )

    return idSysAll, perfAll, CVFitRes


def addMultistepPerfs(
    Y,
    Z,
    methodCodes,
    nxVals,
    U,
    saveFile,
    CVFoldInds,
    YType,
    ZType,
    settings,
    missing_marker,
    idSysAll,
    CVFitRes,
    foldsToRun,
):
    multiStepResultsAreAvailable = True
    steps_ahead = [1, 2, 4, 8, 16, 32]
    MSFieldName = (
        "perfAllMSV2"
        if ("evalMultiStepV2" in settings and settings["evalMultiStepV2"])
        else "perfAllMS"
    )
    if MSFieldName in CVFitRes and not np.array_equal(
        CVFitRes[MSFieldName].shape, CVFitRes["perfAll"].shape
    ):
        logger.warning(
            f"Unexpect dimension for {MSFieldName} (different from perfAll)... discarding"
        )
        del CVFitRes[MSFieldName]
    if MSFieldName in CVFitRes:
        perfAllTestExample = CVFitRes[MSFieldName].flatten()[0]
    elif "perfAll" in CVFitRes:
        perfAllTestExample = CVFitRes["perfAll"].flatten()[0]
    for step_ahead in steps_ahead:
        if f"yCC_{step_ahead}step" not in perfAllTestExample and step_ahead != 1:
            multiStepResultsAreAvailable = False
            break
    logger.info(
        f"Some multistep ahead results for steps_ahead={steps_ahead} are "
        + ("available" if multiStepResultsAreAvailable else "NOT available")
    )
    if (
        (
            ("evalMultiStep" in settings and settings["evalMultiStep"])
            or ("evalMultiStepV2" in settings and settings["evalMultiStepV2"])
        )
        and multiStepResultsAreAvailable
        and MSFieldName not in CVFitRes
    ):
        # Move a copy of perfA to MSFieldName for consistency
        CVFitRes[MSFieldName] = CVFitRes["perfAll"]
    elif (
        ("evalMultiStep" in settings and settings["evalMultiStep"])
        or ("evalMultiStepV2" in settings and settings["evalMultiStepV2"])
    ) and not multiStepResultsAreAvailable:
        import tensorflow as tf

        perfAllMS = np.empty(
            (len(methodCodes), len(foldsToRun), len(nxVals)), dtype=object
        )
        cnt = 0
        Y_in = Y
        if "ySource" in CVFitRes["extraSettings"]:
            inds = []
            for ySrcO in CVFitRes["extraSettings"]["ySource"]:
                inds.append(
                    np.where(
                        [
                            ySrcO["chans"] == ySrc["chans"]
                            and (
                                "cluster" not in ySrcO
                                or ySrcO["cluster"] == ySrc["cluster"]
                            )
                            for ySrc in settings["ySource"]
                        ]
                    )[0]
                )
            inds = np.array(inds)[:, 0]
            if not np.array_equal(inds, np.arange(len(settings["ySource"]))):
                Y_in = Y[:, inds]
                Y_in_Src = settings["ySource"][inds]
                logger.info(
                    "Keeping the same channels as were kept in original results"
                )
        for fi, fNum in enumerate(foldsToRun):
            foldInds = CVFoldInds[fNum - 1]
            (
                YTrain,
                ZTrain,
                YTest,
                ZTest,
                UTrain,
                UTest,
                yMean,
                yStd,
                YTestSc,
                zMean,
                zStd,
                ZTestSc,
                uMean,
                uStd,
                UTestSc,
            ) = prepareTrainingAndTestData(
                Y_in,
                Z,
                foldInds,
                settings,
                U=U,
                YType=YType,
                ZType=ZType,
                missing_marker=missing_marker,
                return_prescaling_copy=False,
            )
            for mi in range(len(methodCodes)):
                methodSettings = fitMethodSettings(methodCodes[mi])
                for nxInd in range(len(nxVals)):
                    cnt += 1
                    # if cnt != 7: continue # TEMP
                    # if nxInd != 5: continue # TEMP
                    logger.info(
                        f"Working on computing mutli-step ahead perf for case {cnt}/{len(foldsToRun)*len(methodCodes)*len(nxVals)} ({methodSettings.methodCode})"
                    )
                    thisPerf = CVFitRes["perfAll"][
                        mi, fi, nxInd
                    ]  # Test set performance, just in case needed

                    sId = copy.deepcopy(idSysAll[mi, fi, nxInd])
                    if hasattr(sId, "restoreModels"):
                        sId.restoreModels()

                    tic = time.perf_counter()
                    if isinstance(sId, LSSM):
                        sId.steps_ahead = steps_ahead
                    # elif isinstance(sId, PSIDModel):
                    #     sId.steps_ahead = steps_ahead
                    #     sId.model.steps_ahead = steps_ahead
                    elif (
                        hasattr(sId, "model1")
                        and hasattr(sId.model1, "steps_ahead")
                        and sId.model1.steps_ahead is None
                    ) or (
                        hasattr(sId, "model2")
                        and hasattr(sId.model2, "steps_ahead")
                        and sId.model2.steps_ahead is None
                    ):
                        # Model was trained without multistep
                        if (
                            "evalMultiStepV2" in settings
                            and settings["evalMultiStepV2"]
                        ):
                            # Approach 2: feeding predicted outputs back as observed outputs for forward prediction
                            sId.set_multi_step_with_data_gen(True)
                            sId.set_steps_ahead(
                                steps_ahead, update_rnn_model_steps=False
                            )
                        elif "evalMultiStep" in settings and settings["evalMultiStep"]:
                            # Approach 1: just using A_KC (mathematically wrong)
                            sId.set_multi_step_with_A_KC(
                                True
                            )  # Enable multistep ahead with A_KC, not a separate A param
                            sId.set_steps_ahead(steps_ahead)
                    else:
                        raise (
                            Exception(
                                f'Forecasting not supported for "{methodSettings.methodCode}"'
                            )
                        )
                        # perfTest0, zPredTest0, yPredTest0, xPredTest0 = evalSysId(sId, YTest, ZTest, UTest=UTest, YType=YType, ZType=ZType, useXFilt=settings['useXFilt'], missing_marker=missing_marker)

                    evalTimeTest = time.perf_counter() - tic

                    sId.enable_forward_pred = True
                    # sId.finetune(transposeIf(YTrain), transposeIf(ZTrain), U=transposeIf(UTrain), epochs=0, verbose=False)
                    perfTest, zPredTest, yPredTest, xPredTest = evalSysId(
                        sId,
                        YTest,
                        ZTest,
                        UTest=UTest,
                        YType=YType,
                        ZType=ZType,
                        useXFilt=settings["useXFilt"],
                        missing_marker=missing_marker,
                    )

                    # sId.finetune(transposeIf(YTrain), transposeIf(ZTrain), U=transposeIf(UTrain), epochs=1000, verbose=True,
                    #     trainableParams={'base': False, 'fw': True})
                    # perfTest2, zPredTest2, yPredTest2, xPredTest2 = evalSysId(sId, YTest, ZTest, UTest=UTest, YType=YType, ZType=ZType, useXFilt=settings['useXFilt'], missing_marker=missing_marker)

                    logger.info(
                        f"Case {cnt}/{len(foldsToRun)*len(methodCodes)*len(nxVals)} done (test re-eval took {evalTimeTest:.1f}s)"
                    )
                    # Double check that the test perfs are the same as computed before
                    checkNewVsOldPerfs(thisPerf, perfTest, YTest, skip_missing=True)
                    perfAllMS[mi, fi, nxInd] = perfTest

                    """
                        # Compute autocorrelation of data
                        import scipy
                        corr_mode = 'full'
                        lags = scipy.signal.correlation_lags(YTest.shape[0], YTest.shape[0], mode=corr_mode)
                        lag0Ind = np.where( lags==0 )[0]
                        computeAutoCorr = lambda YThis: np.array([scipy.signal.correlate(YThis[:,yi:(yi+1)], YThis[:, yi:(yi+1)], mode=corr_mode, method='auto') for yi in range(YThis.shape[1])])[..., 0].T / YThis.shape[0]
                        yCorr = computeAutoCorr(YTest)
                        yCorr = yCorr / yCorr[lag0Ind, :] * np.diag(np.corrcoef(YTest.T))
                        zCorr = computeAutoCorr(ZTest)
                        zCorr = zCorr / zCorr[lag0Ind, :] * np.diag(np.corrcoef(ZTest.T))
                        perfD = {}
                        for step_ahead in steps_ahead:
                            perfDThis = evaluateDecoding(
                                ZTest     = ZTest[step_ahead:, :], 
                                zPredTest = ZTest[:(-step_ahead), :], 
                                YTest     = YTest[step_ahead:, :], 
                                yPredTest = YTest[:(-step_ahead), :], 
                                measures=['CC', 'R2'])
                            for key, val in perfDThis.items():
                                new_key = key if step_ahead == 1 else f'{key}_{step_ahead}step'
                                perfD[ new_key ] = val
                        naivePerfs = perfD
                        for perfM in ['meanCC', 'meanyCC', 'meanR2', 'meanyR2']:
                            perfValsSteps = []
                            perfValsSteps2 = []
                            naivePerfsSteps = []
                            for step_ahead in steps_ahead:
                                perfValsSteps.append( perfTest[perfM+(f'_{step_ahead}step' if step_ahead > 1 else '')] )
                                perfValsSteps2.append( perfTest2[perfM+(f'_{step_ahead}step' if step_ahead > 1 else '')] )
                                naivePerfsSteps.append( naivePerfs[perfM+(f'_{step_ahead}step' if step_ahead > 1 else '')] )
                            plt.figure()
                            plt.plot(steps_ahead, perfValsSteps, label='before finetuning')
                            plt.plot(steps_ahead, perfValsSteps2, label='after finetuning')
                            plt.plot(steps_ahead, naivePerfsSteps, label='naive performance')
                            lagInds = np.where( np.logical_and( lags >= np.min(steps_ahead), lags <= np.max(steps_ahead) ) )[0]
                            if perfM.replace('mean', '')[0] == 'y':
                                corrMean = np.mean(yCorr, axis=1)
                                plt.plot(lags[lagInds], corrMean[lagInds], label='Neural autocorrelation')
                            else:
                                corrMean = np.mean(zCorr, axis=1)
                                plt.plot(lags[lagInds], corrMean[lagInds], label='Behavior autocorrelation')
                            plt.xlabel('Steps ahead')
                            plt.ylabel(perfM)
                            plt.legend()
                        #"""
                    pass
        CVFitRes[MSFieldName] = perfAllMS
        logger.info(
            f"Saving updated results with the multistep ahead test performance measures in {saveFile}"
        )
        saveFileBase = os.path.splitext(saveFile)[0]
        saveCVFitResults(CVFitRes, saveFileBase + "_perftr.p", saveFile)
        try:
            os.rename(saveFile, saveFileBase + "_BUOldPerf.p")
        except:
            logger.warning(
                "{} already exists, deleting {}".format(
                    saveFileBase + "_BUOldPerf.p", saveFile
                )
            )
            os.remove(saveFile)
            pass
        os.rename(saveFileBase + "_perftr.p", saveFile)
    return CVFitRes


def checkNewVsOldPerfs(oldPerf, newPerf, YTest, skip_missing=False):
    oldPerf = sanitize_perfs(oldPerf)[0].flatten()[0]
    newPerf = sanitize_perfs(newPerf)[0].flatten()[0]
    if isinstance(YTest, (list, tuple)):
        YTestCat = np.concatenate(YTest)
        isFlatY = (np.max(YTestCat, axis=0) - np.min(YTestCat, axis=0)) == 0
    else:
        isFlatY = (np.max(YTest, axis=0) - np.min(YTest, axis=0)) == 0
    for pm in newPerf:
        if (pm not in oldPerf or pm not in newPerf) and skip_missing:
            continue
        if (
            "NRMSE" in pm and pm in oldPerf and np.all(np.isnan(oldPerf[pm]))
        ):  # Computation of NRMSE in very old data sometimes produced NaNs; this is ok
            continue
        if np.any(isFlatY) and pm in ["yCC", "yNRMSE", "yEV", "yR2"]:
            newPerf[pm][isFlatY] = (
                1 if "NRMSE" in pm else 0
            )  # These used to be computed as 0 for flat channels
            oldPerf[pm][isFlatY] = (
                1 if "NRMSE" in pm else 0
            )  # Fix any metric that blew up for a flat channel
            newPerf["mean" + pm] = np.mean(newPerf[pm])
            oldPerf["mean" + pm] = np.mean(oldPerf[pm])
        mean_pm = "mean" + pm
        if ("CC" in mean_pm or "R2" in mean_pm or "EV" in mean_pm) and (
            (mean_pm in oldPerf and oldPerf[mean_pm] < 0.02)
            or (pm in oldPerf and np.all(oldPerf[pm] < 0.02))
        ):
            atol = 1e-2
        else:
            atol = 1e-10
        try:
            np.testing.assert_allclose(oldPerf[pm], newPerf[pm], rtol=0.05, atol=1e-10)
        except Exception as e:
            if (
                mean_pm in oldPerf
            ):  # We mostly care about mean metrics over data dimensions. Skip if that is fine.
                if np.all(oldPerf[mean_pm] > 1e30) and np.all(
                    np.isnan(newPerf[mean_pm])
                ):
                    logger.warning(
                        f"We are dealing with a blow-up that remains a blow-up. Letting it pass..."
                    )
                elif pm in ["CM", "meanCM"]:
                    logger.info(f"Skipping exact checking of ConfusionMatrix metric")
                else:
                    np.testing.assert_allclose(
                        oldPerf[mean_pm], newPerf[mean_pm], rtol=0.05, atol=atol
                    )
                logger.warning(
                    f"There was some minor discrepency in {pm} (see below) but {mean_pm} was close, so skipping: {e}"
                )
            elif np.all(oldPerf[pm] > 1e30) and np.all(np.isnan(newPerf[pm])):
                logger.warning(
                    f"We are dealing with a blow-up that remains a blow-up. Letting it pass..."
                )
            elif pm not in ["CM", "meanCM"]:
                rtol = (
                    0.25
                    if ("CC" in pm or "R2" in pm or "EV" in pm) and oldPerf[pm] < 0.02
                    else 0.05
                )
                np.testing.assert_allclose(
                    oldPerf[pm], newPerf[pm], rtol=0.05, atol=atol
                )
        isClose = np.isclose(oldPerf[pm], newPerf[pm], rtol=1e-6, atol=1e-10)
        isKindOfClose = np.isclose(oldPerf[pm], newPerf[pm], rtol=1e-2, atol=1e-2)
        areNaNs = np.logical_and(np.isnan(oldPerf[pm]), np.isnan(newPerf[pm]))
        if np.all(areNaNs):
            logger.warning(
                f'"{pm}" is nan in all dimensions, probably due to flat or unstable model! Moving on since this has been the case in original results as well. '
            )
        isClose = np.logical_or(isClose, areNaNs)
        if not isinstance(newPerf[pm], np.ndarray) or len(newPerf[pm]) == 1:
            isClose = np.array([isClose])
        if not np.all(isClose):
            logger.warning(
                f'WARNING: There is some difference between the original and recomputed test performance for "{pm}" (in {np.sum(~isClose)}/{len(isClose)} dimensions)'
            )
        if np.all(~isClose) and len(isClose) > 1:
            if np.sum(isKindOfClose) / len(isKindOfClose) >= 0.5:
                logger.warning(
                    f'"{pm}" is not exactly what was computed before, but over 50% of dimensions ({np.sum(isKindOfClose)} of {len(isKindOfClose)}) are sort of close (within 1% of previous values), so we will let it pass'
                )
            elif "SNR" in pm:
                logger.warning(f'"{pm}" is not important... skipping')
            else:
                raise (Exception("Something is too off!"))
    return True


def addTrainingPerformanceIfMissing(
    Y,
    Z,
    methodCodes,
    nxVals,
    U,
    T,
    saveFile,
    CVFoldInds,
    YType,
    ZType,
    settings,
    missing_marker,
    idSysAll,
    perfAll,
    CVFitRes,
    CVFolds=None,
    foldsToRun=None,
):
    folds = np.arange(1, 1 + CVFolds)
    if len(settings["foldsToRun"]) < 1:
        settings["foldsToRun"] = np.copy(folds)
    if foldsToRun is None:
        foldsToRun = settings["foldsToRun"]
    buggyTrainingPerf = False
    modTime = datetime.fromtimestamp(os.path.getmtime(saveFile))
    if "perfAllTrain" in CVFitRes and modTime <= datetime(
        2024, 4, 25, 0, 0, 0
    ):  # We are dealing with result that may have buggy training perf (from when training data was scaled before being passed to evalIdSys below)
        buggyTrainingPerf = True
    if buggyTrainingPerf:
        saveFileBase = os.path.splitext(saveFile)[0]
        saveFileBU = saveFileBase + "_BUBuggyTrainPerf.p"
        shutil.copy2(saveFile, saveFileBU)
        logger.warning(
            f"Results file is from before the bug in computing training perf was fixed. Will back this up (as {saveFileBU}) and recompute training perf."
        )
        del CVFitRes["perfAllTrain"]
    if "perfAllTrain" in CVFitRes and (
        not np.array_equal(CVFitRes["perfAllTrain"].shape, CVFitRes["perfAll"].shape)
        or np.any(np.vectorize(lambda a: a is None)(CVFitRes["perfAllTrain"]))
    ):
        logger.warning(
            "Unexpect dimension for (or None entry found in) perfAllTrain (different from perfAll)... discarding"
        )
        del CVFitRes["perfAllTrain"]
    if (
        "computePerfInTraining" in settings
        and settings["computePerfInTraining"]
        and "perfAllTrain" in CVFitRes
    ):
        logger.info(
            "perfAllTrain is included in loaded results... no need to recompute."
        )
    if (
        "computePerfInTraining" in settings
        and settings["computePerfInTraining"]
        and "perfAllTrain" not in CVFitRes
    ):
        import tensorflow as tf

        perfAllTestRecomputed = np.empty(
            (len(methodCodes), len(foldsToRun), len(nxVals)), dtype=object
        )
        perfAllTrain = np.empty(
            (len(methodCodes), len(foldsToRun), len(nxVals)), dtype=object
        )
        cnt = 0
        Y_in = Y
        if "ySource" in CVFitRes["extraSettings"]:
            inds = []
            for ySrcO in CVFitRes["extraSettings"]["ySource"]:
                inds.append(
                    np.where(
                        [
                            (
                                ("chans" in ySrcO and ySrcO["chans"] == ySrc["chans"])
                                or ("chan" in ySrcO and ySrcO["chan"] == ySrc["chan"])
                            )
                            and (
                                "bandInd" not in ySrcO
                                or ySrcO["bandInd"] == ySrc["bandInd"]
                            )  # For powers
                            and (
                                "cluster" not in ySrcO
                                or ySrcO["cluster"] == ySrc["cluster"]
                            )  # For sorted spiking data
                            for ySrc in settings["ySource"]
                        ]
                    )[0]
                )
            inds = np.array(inds)[:, 0]
            if not np.array_equal(inds, np.arange(len(settings["ySource"]))):
                Y_in = Y[:, inds]
                Y_in_Src = settings["ySource"][inds]
                logger.info(
                    "Keeping the same channels as were kept in original results"
                )

        error_cases = []
        for fi, fNum in enumerate(foldsToRun):
            foldInds = CVFoldInds[fNum - 1]
            (
                YTrain,
                ZTrain,
                YTest,
                ZTest,
                UTrain,
                UTest,
                yMean,
                yStd,
                YTestSc,
                zMean,
                zStd,
                ZTestSc,
                uMean,
                uStd,
                UTestSc,
                YTrainPreScaling,
                ZTrainPreScaling,
                UTrainPreScaling,
            ) = prepareTrainingAndTestData(
                Y_in,
                Z,
                foldInds,
                settings,
                U=U,
                YType=YType,
                ZType=ZType,
                missing_marker=missing_marker,
                return_prescaling_copy=True,
            )
            for mi in range(len(methodCodes)):
                methodCode = methodCodes[mi]
                methodSettings = fitMethodSettings(methodCode)
                for nxInd in range(len(nxVals)):
                    cnt += 1
                    logger.info(
                        f"Working on computing perf in training data for case {cnt}/{len(foldsToRun)*len(methodCodes)*len(nxVals)} (fold: {fNum}, nx: {nxVals[nxInd]})"
                    )
                    thisPerf = CVFitRes["perfAll"][
                        mi, fi, nxInd
                    ]  # Test set performance, just in case needed

                    sId = copy.deepcopy(idSysAll[mi, fi, nxInd])

                    if hasattr(sId, "restoreModels"):
                        sId.restoreModels()
                    tic = time.perf_counter()
                    perfTrain, zPredTrain, yPredTrain, xPredTrain = evalSysId(
                        sId,
                        YTrainPreScaling,
                        ZTrainPreScaling,
                        UTest=UTrainPreScaling,
                        YType=YType,
                        ZType=ZType,
                        useXFilt=settings["useXFilt"],
                        missing_marker=missing_marker,
                    )
                    evalTimeTraining = time.perf_counter() - tic
                    tic = time.perf_counter()
                    perfTest, zPredTest, yPredTest, xPredTest = evalSysId(
                        sId,
                        YTest,
                        ZTest,
                        UTest=UTest,
                        YType=YType,
                        ZType=ZType,
                        useXFilt=settings["useXFilt"],
                        missing_marker=missing_marker,
                    )
                    evalTimeTest = time.perf_counter() - tic
                    logger.info(
                        f"Case {cnt}/{len(foldsToRun)*len(methodCodes)*len(nxVals)} done (training eval took {evalTimeTraining:.1f}s and test re-eval took {evalTimeTest:.1f}s)"
                    )
                    # Double check that the test perfs are the same as computed before
                    try:
                        perfAllTestRecomputed[mi, fi, nxInd] = perfTest
                        perfAllTrain[mi, fi, nxInd] = perfTrain
                        checkNewVsOldPerfs(thisPerf, perfTest, YTest, skip_missing=True)
                        # if 'meanR2' in perfTest and 0.75*perfTest['meanR2'] > perfTrain['meanR2']:
                        #     raise(Exception('Not expected!'))
                        # if 'meanAUC' in perfTest and 0.75*perfTest['meanAUC'] > perfTrain['meanAUC']:
                        #     raise(Exception('Not expected!'))
                    except Exception as e:
                        logger.error(e)
                        error_cases.append(
                            {
                                "nx": nxVals[nxInd],
                                "fold": fNum,
                                "methodCode": methodCode,
                                "nxInd": nxInd,
                                "fi": fi,
                                "mi": mi,
                                "perfTestOld": thisPerf,
                                "perfTestNew": perfTest,
                            }
                        )

                        # raise(Exception(e))
                    """
                        # Plot predictions
                        _, zPredTestOG, yPredTestOG, xPredTestOG = evalSysId(sId, YTest, ZTest, UTest=UTest, YType=YType, ZType=ZType, useXFilt=settings['useXFilt'], missing_marker=missing_marker, undo_scaling=True)
                        figBasePath = os.path.join(os.path.dirname(saveFile), 'Figs', os.path.basename(saveFile)+f'_f{fNum}_nx{nxVals[nxInd]}_mi{mi}')
                        plotArgs = {
                            'y_pred_is_list': True, 'missing_marker': missing_marker, 'predPerfsToAdd': ['R2', 'CC'], 'saveExtensions': ['png'], 'lineStyles':['-', '--', '-.', ':'],
                            'predLegStrs': None if (not hasattr(sId, 'steps_ahead') or sId.steps_ahead is None) else [f'{sa}-step Pred' for sa in sId.steps_ahead],
                            'XLim': np.array([0,np.min([2000, ZTest.shape[0]])]),
                        }
                        titleBase = f'{methodCodes[mi]}, nx={nxVals[nxInd]}, Fold {fNum}'
                        plotTimeSeriesPrediction(ZTest, zPredTestOG, title=titleBase + ', test z pred', saveFile=figBasePath+'_testZPred', **plotArgs)
                        plotTimeSeriesPrediction(YTest, yPredTestOG, title=titleBase + ', test y pred', saveFile=figBasePath+'_testYPred', **plotArgs)
                        """
                    pass
            # pickle_save(tmp_save, {'perfAllTestRecomputed': perfAllTestRecomputed, 'perfAllTrain': perfAllTrain, 'error_cases': error_cases})
        CVFitRes["perfAllTrain"] = perfAllTrain
        if (
            len(error_cases) > 0 and len(error_cases) < 0.2 * perfAllTrain.size
        ):  # Very few cases have updated test perf.
            # These are most likely cases were the orignal fit was redone and overwritten by a new job on the server after
            # the aggregate results were saved. It is ok to replace these test results with recomputed test results.
            logger.warning(
                f"Updating test perfs in {len(error_cases)}/{perfAllTrain.size} cases with recomputed values to be consistent with final saved fitted files on disk."
            )
            CVFitRes["perfAllBU"] = copy.deepcopy(CVFitRes["perfAll"])
            CVFitRes["updatedCases"] = error_cases
            perfAllTest = copy.deepcopy(CVFitRes["perfAll"])
            for error_case in error_cases:
                nxInd = error_case["nxInd"]
                fi = error_case["fi"]
                mi = error_case["mi"]
                logger.info(
                    f"Updating test perfs for (fold: {fNum}, nx: {nxVals[nxInd]})"
                )
                perfAllTest[mi, fi, nxInd] = perfAllTestRecomputed[mi, fi, nxInd]
            CVFitRes["perfAll"] = perfAllTest
            perfAll = perfAllTest
        elif len(error_cases) > 0:
            print(
                "Too much difference, if you really want to go ahead with replacing test perfs with recomputed values, run the next few lines manually at a breakpoint."
            )
        CVFitRes["updatedCases"] = error_cases
        logger.info(
            f"Saving updated results with the Training performance measures in {saveFile}"
        )
        saveFileBase = os.path.splitext(saveFile)[0]
        saveCVFitResults(CVFitRes, saveFileBase + "_perftr.p", saveFile)
        try:
            os.rename(saveFile, saveFileBase + "_BUOldPerf.p")
        except:
            logger.warning(
                "{} already exists, deleting {}".format(
                    saveFileBase + "_BUOldPerf.p", saveFile
                )
            )
            os.remove(saveFile)
            pass
        os.rename(saveFileBase + "_perftr.p", saveFile)
    return perfAll, CVFitRes


def doCVedModelingForGivenFolds(
    Y,
    Z,
    U,
    methodCodes,
    nxVals,
    n1Vals,
    trueSys,
    CVFolds,
    CVFoldInds,
    foldsToRun,
    methodIndsToRun,
    T,
    YType,
    ZType,
    UType,
    ZClasses,
    settings,
    missing_marker,
    saveFile,
):
    """Performs CVed cross-validation for the specified folds

    Args:
        Y (np.ndarray): signal 1 (e.g. neural activity)
        Z (np.ndarray): signal 2 (e.g. behavior)
        U (np.ndarray): input (e.g. task instructions)
        methodCodes (string): method code
        nxVals (list of int): state dimension nx values to fit
        n1Vals (list of int): state dimension in stage 1 (n1) value(s) to use
        trueSys ([type]): object for true mode (used for 'Ideal' eval)
        CVFolds (int): number of CV folds
        CVFoldInds (list of dict): train/test indices of the folds
        foldsToRun (list of int): folds to run
        methodIndsToRun (list of int): method indexes to run
        T (np.ndarray): time vector. size must be consistent with Y,Z,U
        YType (string): data type of Y
        ZType (string): data type of Z
        UType (string): data type of U
        ZClasses (np.array): class values in Z
        settings (dict): dict with the run settings
        missing_marker (number): the marker value for missing data
        saveFile ([type]): path of save file

    Returns:
        idSysAll (object): learned models
        perfAll (dict): performance measures
        CVFitRes (dict): model fitting results
    """
    foldSaveFiles = ["" for fi in range(len(foldsToRun))]
    perfAll = np.empty((len(methodCodes), len(foldsToRun), len(nxVals)), dtype=object)
    idSysAll = np.empty((len(methodCodes), len(foldsToRun), len(nxVals)), dtype=object)
    idInfoAll = np.empty((len(methodCodes), len(foldsToRun), len(nxVals)), dtype=object)
    isTrialBased = isinstance(Y, (list, tuple))
    if not isTrialBased:
        yPredAll, zPredAll = preparePredictionPlaceHolders(
            Y, Z, YType, ZType, ZClasses, methodCodes, nxVals
        )
    else:
        zPredAll, yPredAll = [], []
        for ti in range(len(Y)):
            yPredAllThis, zPredAllThis = preparePredictionPlaceHolders(
                Y[ti],
                Z[ti] if Z is not None else None,
                YType,
                ZType,
                ZClasses,
                methodCodes,
                nxVals,
            )
            zPredAll.append(zPredAllThis)
            yPredAll.append(yPredAllThis)

    fullyDoneAll = [True for fi in range(len(foldsToRun))]

    if -1 in foldsToRun:
        foldsToRun = list(range(len(CVFoldInds)))
        logger.info(
            f"foldsToRun has -1, assuming all folds need to run, so taking foldsToRun={foldsToRun}"
        )

    for fi, fNum in enumerate(foldsToRun):
        # Prepare CV folds
        foldInds = CVFoldInds[fNum - 1]
        trainInds = foldInds["trainInds"]
        testInds = foldInds["testInds"]
        isCloseInd = None
        if "isCloseInd" in foldInds:
            isCloseInd = foldInds["isCloseInd"]
        logger.info(
            f"Fold {fNum}/{len(CVFoldInds)}: training samples: {len(trainInds)}, test samples={len(testInds)}"
        )

        foldRes, foldSaveFilePath = loadFoldResults(
            saveFile, fNum, CVFoldInds, settings
        )
        foldSaveFiles[fi] = foldSaveFilePath
        if foldRes is None:
            foldRes, nxVals, fullyDoneAll[fi] = doCVedModelingForFold(
                Y,
                Z,
                U,
                methodCodes,
                nxVals,
                n1Vals,
                trueSys,
                CVFolds,
                CVFoldInds,
                fNum,
                methodIndsToRun,
                T,
                YType,
                ZType,
                UType,
                ZClasses,
                settings,
                missing_marker,
                foldSaveFilePath,
                saveFile,
            )

        if not fullyDoneAll[fi]:
            continue
        # Combine results from folds.
        for mi, methodCode in enumerate(methodCodes):
            for nxi, nx in enumerate(nxVals):
                idSys = foldRes[mi][nxi]["idSys"]
                idSysAll[mi, fi, nxi] = idSys
                if "idInfo" in foldRes[mi][nxi]:
                    idInfoAll[mi, fi, nxi] = foldRes[mi][nxi]["idInfo"]
                perfAll[mi, fi, nxi] = foldRes[mi][nxi]["perf"]
        # Combine predictions acrpss folds
        if not isTrialBased:
            yPredAll, zPredAll = combinePredictionsAcrossFolds(
                foldInds,
                foldRes,
                fi,
                fNum,
                yPredAll,
                zPredAll,
                YType,
                ZType,
                methodCodes,
                nxVals,
                missing_marker,
            )
        else:
            for ti in testInds:
                yPredAll[ti], zPredAll[ti] = combinePredictionsAcrossFolds(
                    foldInds,
                    foldRes,
                    fi,
                    fNum,
                    yPredAll[ti],
                    zPredAll[ti],
                    YType,
                    ZType,
                    methodCodes,
                    nxVals,
                    missing_marker,
                    trialInd=ti,
                )

    if not np.all(fullyDoneAll):
        raise (
            Exception(
                "Done running all nxValsToRun={}, but not all nxVals={}".format(
                    settings["nxValsToRun"], nxVals
                )
            )
        )

    if "foldsToConsider" in settings and settings["foldsToConsider"] is not None:
        neededFolds = settings["foldsToConsider"]
    else:
        neededFolds = list(np.arange(1, CVFolds))

    CVFitRes = {}
    if not np.all([f in neededFolds for f in neededFolds]):
        logger.info("Not all folds are done... Skipping the merging of folds...")
        return idSysAll, perfAll, CVFitRes

    catIfNeeded = lambda L, axis=0: (
        np.concatenate(L, axis=axis) if isTrialBased and L is not None else L
    )

    perfAllFCat = np.empty((len(methodCodes), 1, len(nxVals)), dtype=object)
    for mi in range(len(methodCodes)):
        for nxi in range(len(nxVals)):
            if not isTrialBased:
                yPredA = yPredAll[mi, nxi]
                zPredAO = np.array(zPredAll[mi, nxi])
            else:
                yPredA = np.concatenate([yPred[mi, nxi] for yPred in yPredAll])
                zPredAO = (
                    np.concatenate([np.array(zPred[mi, nxi]) for zPred in zPredAll])
                    if zPredAll[0] is not None
                    else None
                )
            if (
                np.sum(
                    np.logical_and(
                        np.any(~np.isnan(yPredA), axis=1),
                        np.any(yPredA != np.nan, axis=1),
                    )
                )
                == 0
            ):
                yPredA = None
            if (
                "avgPredsInSampleGroups" in settings
                and settings["avgPredsInSampleGroups"]
            ):
                zPredA = np.ones(zPredAO.shape)
                zPredA[:] = np.nan
                for gInds in settings["avgPredSampleGroups"]:
                    if gInds.size > 0:
                        zPredMean = np.mean(zPredAO[gInds, :], axis=0)
                        zPredA[gInds[-1], :] = zPredMean
            else:
                zPredA = zPredAO
            perfAllFCat[mi, 0, nxi] = evaluateDecoding(
                ZTest=catIfNeeded(Z),
                zPredTest=zPredA,
                YTest=catIfNeeded(Y),
                yPredTest=yPredA,
                YType=YType,
                ZType=ZType,
                missing_marker=missing_marker,
            )

    # if saveFile is not None and Z is not None:
    #     plot_CVed_decoding(catIfNeeded(Z), T, ZType, settings, missing_marker, catIfNeeded(zPredAll,2), perfAllFCat, saveFile=saveFile)

    if "savePreds" not in settings:
        settings["savePreds"] = False
    if "doNotSaveMissing" not in settings:
        settings["doNotSaveMissing"] = False
    if "removeFoldResults" not in settings:
        settings["removeFoldResults"] = True

    CVFitRes["extraSettings"] = settings
    CVFitRes["CVFoldInds"] = CVFoldInds
    CVFitRes["idSysAll"] = np.array(idSysAll)
    CVFitRes["idInfoAll"] = np.array(idInfoAll)
    CVFitRes["perfAll"] = np.array(perfAll)
    CVFitRes["perfAllFCat"] = perfAllFCat

    if settings["savePreds"]:
        CVFitRes["zPredAllUnit"] = (
            "z"  # Any zscoring is undone so that zPredAll is in the same unit as z
        )
        if settings["doNotSaveMissing"] is False:
            CVFitRes["zPredAll"] = zPredAll
            CVFitRes["z"] = {"data": Z, "time": T}
        else:
            notMissing = ~np.any(Z == missing_marker, axis=1)
            zPredAllNotM = zPredAll[:, :, notMissing, :]
            CVFitRes["notMissing"] = notMissing
            CVFitRes["zPredAll"] = zPredAllNotM
            CVFitRes["z"] = {"data": Z[notMissing, :], "time": None}
            if T is not None:
                CVFitRes["z"]["time"] = T[notMissing]

    if saveFile is not None:
        saveCVFitResults(CVFitRes, saveFile)
        if "removeFoldResults" in settings and settings["removeFoldResults"]:
            removeFoldResultFiles(methodCodes, nxVals, CVFolds, foldSaveFiles)

    return idSysAll, perfAll, CVFitRes


def preparePredictionPlaceHolders(Y, Z, YType, ZType, ZClasses, methodCodes, nxVals):
    if Z is not None:
        if YType == "cat":
            zPredAll = np.empty(
                (len(methodCodes), len(nxVals), Z.shape[0], Z.shape[1], len(ZClasses)),
                dtype=float,
            )
            zPredAll[:] = np.nan
        else:  # 'cont' or 'count_process'
            zPredAll = np.empty(
                (len(methodCodes), len(nxVals), Z.shape[0], Z.shape[1]), dtype=Z.dtype
            )
            zPredAll[:] = np.nan
    else:
        zPredAll = None
    yPredAll = np.empty(
        (len(methodCodes), len(nxVals), Y.shape[0], Y.shape[1]), dtype=Y.dtype
    )
    yPredAll[:] = np.nan
    return yPredAll, zPredAll


def combinePredictionsAcrossFolds(
    foldInds,
    foldRes,
    fi,
    fNum,
    yPredAll,
    zPredAll,
    YType,
    ZType,
    methodCodes,
    nxVals,
    missing_marker,
    trialInd=None,
):
    isTrain = foldInds["trainInds"]
    isTest = foldInds["testInds"]
    isCloseInd = None
    if "isCloseInd" in foldInds:
        isCloseInd = foldInds["isCloseInd"]

    # isNonNan are values that are still missing and need to be populated.
    if zPredAll is not None:
        if ZType == "cat":
            isNonNan = np.nonzero(np.all(np.isnan(zPredAll[0, 0, :, :, :]), axis=1))[0]
        else:
            isNonNan = np.nonzero(np.all(np.isnan(zPredAll[0, 0, :, :]), axis=1))[0]
    else:
        if YType == "cat":
            isNonNan = np.nonzero(np.all(np.isnan(yPredAll[0, 0, :, :, :]), axis=1))[0]
        else:
            isNonNan = np.nonzero(np.all(np.isnan(yPredAll[0, 0, :, :]), axis=1))[0]

    if trialInd is None:  # Not trial based
        isTestNonNan = np.isin(isTest, isNonNan)  # Must be in test data
        if isCloseInd is not None:
            # We force using results from the most recent fold for isCloseInd indices -- if provided.
            isTestNonNan = np.logical_or(isTestNonNan, np.isin(isTest, isCloseInd))
        isTestNN = isTest[isTestNonNan]
        if np.any(~isTestNonNan):
            logger.warning(
                "WARNING: There is {} samples of overlap (our of {}) between test data in fold {} and prior folds. Will skip these samples and only aggregate test data from the new parts of this test fold.".format(
                    len(isTest) - len(isTestNN), len(isTest), fNum
                )
            )
            if len(isTestNN) == 0:
                logger.warning(
                    "WARNING: no samples in the test set are distinct from all prior test sets! Will ignore test set predictions from this fold."
                )
            """
                sns.scatterplot(x=T[Z[:, 0]!=missing_marker], y=Z[Z[:, 0]!=missing_marker, 0])
                sns.scatterplot(x=T[isTest], y=np.ones(T[isTest].shape))
                sns.scatterplot(x=T[isTestNN], y=np.ones(T[isTestNN].shape))
                plt.show()
                """
    else:
        isTestNonNan = (np.ones_like(isNonNan) == 1) * (
            trialInd in isTest
        )  # Must be in test data
        isTestNN = isTestNonNan

    for mi, methodCode in enumerate(methodCodes):
        for nxi, nx in enumerate(nxVals):
            idSys = foldRes[mi][nxi]["idSys"]
            zPredTest = foldRes[mi][nxi]["zPredTest"]
            yPredTest = foldRes[mi][nxi]["yPredTest"]
            if (
                trialInd is not None
            ):  # Multi-step and trial-based (isinstance(zPredTest, list) and isinstance(zPredTest[0], list))
                thisTestTrialIndexWithinFold = np.where(isTest == trialInd)[0][0]
                zPredTest = zPredTest[
                    thisTestTrialIndexWithinFold
                ]  # In case of multistep ahead, keep the first
                yPredTest = yPredTest[
                    thisTestTrialIndexWithinFold
                ]  # In case of multistep ahead, keep the first
            if isinstance(zPredTest, list):
                zPredTest = zPredTest[0]  # In case of multistep ahead, keep the first
            if zPredTest is not None:
                if ZType != "cat":
                    if ZType == "cont":
                        zPredAll[mi, nxi, isTestNN, :] = undoScaling(
                            idSys,
                            zPredTest[isTestNonNan, :],
                            "zMean",
                            "zStd",
                            missing_marker=missing_marker,
                        )
                    else:
                        zPredAll[mi, nxi, isTestNN, :] = zPredTest[isTestNonNan, :]
                else:
                    zPredAll[mi, nxi, isTestNN, :, :] = zPredTest[isTestNonNan, :, :]
            if isinstance(yPredTest, list):
                yPredTest = yPredTest[0]  # In case of multistep ahead, keep the first
            if type(yPredTest) is np.ndarray:
                if YType != "cat":
                    if YType == "cont":
                        yPredAll[mi, nxi, isTestNN, :] = undoScaling(
                            idSys,
                            yPredTest[isTestNonNan, :],
                            "yMean",
                            "yStd",
                            missing_marker=missing_marker,
                        )
                    else:
                        yPredAll[mi, nxi, isTestNN, :] = yPredTest[isTestNonNan, :]
                else:
                    yPredAll[mi, nxi, isTestNN, :, :] = yPredTest[isTestNonNan, :, :]
    return yPredAll, zPredAll


def doCVedModelingForFold(
    Y,
    Z,
    U,
    methodCodes,
    nxVals,
    n1Vals,
    trueSys,
    CVFolds,
    CVFoldInds,
    fNum,
    methodIndsToRun,
    T,
    YType,
    ZType,
    UType,
    ZClasses,
    settings,
    missing_marker,
    foldSaveFilePath,
    saveFile,
    min_cpu_for_parallel=40,
):
    """performs CVed model fitting and evaluation for one fold

    Args:
        Y (np.ndarray): signal 1 (e.g. neural activity)
        Z (np.ndarray): signal 2 (e.g. behavior)
        U (np.ndarray): input (e.g. task instructions)
        methodCodes (string): method code
        nxVals (list of int): state dimension nx values to fit
        n1Vals (list of int): state dimension in stage 1 (n1) value(s) to use
        trueSys ([type]): object for true mode (used for 'Ideal' eval)
        CVFolds (int): number of CV folds
        CVFoldInds (list of dict): train/test indices of the folds
        fNum (int): fold number to fit
        methodIndsToRun (list of int): list of method indexes to run
        T (np.ndarray): time vector. size must be consistent with Y,Z,U
        YType (string): data type of Y
        ZType (string): data type of Z
        UType (string): data type of U
        ZClasses (np.array): class values in Z
        settings (dict): dict with the run settings
        missing_marker (number): the marker value for missing data
        foldSaveFilePath (string): path to save file for fold
        min_cpu_for_parallel (int): minimum number of cpus that if the machine has run over methods will be parallelized

    Returns:
        foldRes (dict): fold results
        nxVals (list): values of nx
    """
    foldInds = CVFoldInds[fNum - 1]

    isTrain = foldInds["trainInds"]
    isTest = foldInds["testInds"]
    if "isCloseInd" in foldInds:  # the time points closest to bev measurement
        isCloseInd = foldInds["isCloseInd"]

    logger.info(
        "Running fold {} (NTrain={}, NTest={})".format(fNum, len(isTrain), len(isTest))
    )

    if np.any(np.isin(isTest, isTrain)):
        logger.warning(
            "WARNING: TESTING IS NOT CROSS VALIDATED. {:.3g}% of the {} sample test data is used among the {} training samples".format(
                np.sum(np.isin(isTest, isTrain)) / len(isTest) * 100,
                len(isTest),
                len(isTrain),
            )
        )

    priorIdSysAll = None
    if len(methodCodes) == 1:
        methodCode = methodCodes[0]
        # Check if a prior learned model that is just different in the readout is already available
        methodSettings = fitMethodSettings(methodCode)
        priorSaveFiles, priorLogStrs = [], []
        if (
            methodSettings.posthocMap is not None
            and methodSettings.supports_posthocMap
            and not methodSettings.keepPostHocMapForICV
        ):
            altPostHocMaps = ["", "KNNR", "RR"]
            for altPostHocMap in altPostHocMaps:
                priorMethodCode = methodCode.replace(
                    methodSettings.posthocMap, altPostHocMap
                )
                priorMethodCode = priorMethodCode.replace("__", "_")
                if priorMethodCode[-1] == "_":
                    priorMethodCode = priorMethodCode[:-1]
                priorSaveFile = saveFile.replace(methodCode, priorMethodCode)
                priorSaveFiles.append(priorSaveFile)
                priorLogStrs.append(
                    f'with different pothocMap: {altPostHocMap} instead of {methodSettings.posthocMap}. We will use the dynamic models from this file and just (re)fit the readout mappings instead of fitting the whole "{methodCode}"'
                )
        elif methodSettings.finetune:
            steps, steps_w, matches = parseMethodCodeArgStepsAhead(methodCode)
            if steps is not None and steps != [1]:
                steps_pattern = "_" + "_".join([match.group() for match in matches])
            else:
                steps_pattern = ""
            if methodSettings.finetuneAgainAfterUnfreeze:
                priorMethodCode = methodCode.replace(
                    "redo_finetuneAAUnf", "redo_finetune"
                ).replace("finetuneAAUnf", "finetune")
                priorSaveFiles.append(saveFile.replace(methodCode, priorMethodCode))
                priorLogStrs.append(
                    f"without finetuning of only forecasting params. We will load models from there and finetune all parameters."
                )
                priorMethodCode = (
                    methodCode.replace("redo_finetuneAAUnf", "")
                    .replace("finetuneAAUnf", "")
                    .replace("__", "_")
                    .replace(steps_pattern, "")
                    .replace("__", "_")
                )
                priorSaveFiles.append(saveFile.replace(methodCode, priorMethodCode))
                priorLogStrs.append(
                    f"without forecasting ability. We will load models from there and finetune them for forecasting. "
                )
            else:
                priorMethodCode = (
                    methodCode.replace("redo_finetuneUnf", "")
                    .replace("redo_finetune", "")
                    .replace("finetuneUnf", "")
                    .replace("finetune", "")
                    .replace("__", "_")
                    .replace(steps_pattern, "")
                    .replace("__", "_")
                )
                priorSaveFiles.append(saveFile.replace(methodCode, priorMethodCode))
                priorLogStrs.append(
                    f"without forecasting ability. We will load models from there and finetune them for forecasting. "
                )
        priorIdSysAll, priorCVFitRes = loadAndParseFullCVedResultsIfExists(
            priorSaveFiles,
            priorLogStrs,
            Z,
            methodCodes,
            CVFolds,
            nxVals,
            CVFoldInds,
            settings,
            missing_marker,
        )

    foldRes = [
        [{} for nxi in range(len(nxVals))] for mi in range(len(methodCodes))
    ]  # method x nx
    foldSaveFilePathMNxAll = [
        [None for nxi in range(len(nxVals))] for mi in range(len(methodCodes))
    ]  # method x nx

    logger.info(
        "Signal dimensions: y => {}, z => {}, u => {}".format(
            Y.shape if isinstance(Y, np.ndarray) else None,
            Z.shape if isinstance(Z, np.ndarray) else None,
            U.shape if isinstance(U, np.ndarray) else None,
        )
    )
    logger.info("Method codes to fit: {}".format(methodCodes))
    logger.info("Nx values to fit: {}".format(nxVals))
    methodCodesEnumeration = list(enumerate(methodCodes))
    exitWhenICVIsDone = False
    # methodCodesEnumeration, exitWhenICVIsDone = reorderMethodsToRun(methodCodesEnumeration)
    if len(methodIndsToRun) > 0:
        methodCodesEnumeration = [
            (mi, methodCode)
            for (mi, methodCode) in methodCodesEnumeration
            if mi in methodIndsToRun
        ]
        exitWhenICVIsDone = True
    argsAll = []
    for mi, methodCode in methodCodesEnumeration:
        mSettings = fitMethodSettings(methodCode)
        preferGPU = "CEBRA" in mSettings.method
        if priorIdSysAll is not None:  # Prior dynamic model fit
            priorSIdAllNx = priorIdSysAll[mi, fNum - 1, :]
            priorIdInfoAllNx = priorCVFitRes["idInfoAll"][mi, fNum - 1, :]
            priorIdSettings = priorCVFitRes["extraSettings"]
        else:
            priorSIdAllNx = None
            priorIdInfoAllNx = None
            priorIdSettings = None
        args = {
            "caseLogStr": 'fold {}/{} method "{}" ({}/{})'.format(
                fNum, CVFolds, methodCode, mi + 1, len(methodCodes)
            ),
            "Y": Y,
            "Z": Z,
            "U": U,
            "T": T,
            "foldInds": foldInds,
            "nxVals": nxVals if mSettings.nx is None else [mSettings.nx],
            "n1Vals": copy.deepcopy(n1Vals) if mSettings.n1 is None else [mSettings.n1],
            "iCVResults": None,
            "methodCode": methodCode,
            "settings": copy.deepcopy(settings),
            "true_model": trueSys,
            "YType": YType,
            "ZType": ZType,
            "UType": UType,
            "ZClasses": ZClasses,
            "missing_marker": missing_marker,
            "priorSIdAllNx": priorSIdAllNx,
            "priorIdInfoAllNx": priorIdInfoAllNx,
            "priorIdSettings": priorIdSettings,
            "CVFoldInds": CVFoldInds,
            "fNum": fNum,
            "saveFilePath": f"{foldSaveFilePath[:-2]}_{methodCode}.p",
            "savePathTail": "",  # f'_{methodCode}'
        }
        argsAll.append(args)

    num_processors = multiprocessing.cpu_count()

    if num_processors < min_cpu_for_parallel or preferGPU:  # To run serially
        results = [doCVedModelingForMethodFoldWithArgs(args) for args in argsAll]
    else:
        logger.info(
            f"Running method-folds in parallel over {num_processors} processors"
        )
        # To run in parallel
        pool = multiprocessing.Pool(
            processes=num_processors - 1
        )  # use all but one of the available processors
        results = pool.map(doCVedModelingForMethodFoldWithArgs, argsAll)
        pool.close()
        pool.join()

    for mi, methodCode in methodCodesEnumeration:
        foldRes[mi], foldSaveFilePathMNxAll[mi] = results[mi]

    if (
        exitWhenICVIsDone
        or (
            "nxValsToRun" in settings
            and len(settings["nxValsToRun"]) > 0
            and nxVals != settings["nxValsToRun"]
        )
        or (
            "nxIndsToRun" in settings
            and len(settings["nxIndsToRun"]) > 0
            and -1 not in settings["nxIndsToRun"]
            and list(range(nxVals)) != settings["nxIndsToRun"]
        )
    ):
        return foldRes, nxVals, False

    if foldSaveFilePath is not None:
        # Save fold results:
        Path(foldSaveFilePath).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving results for fold {} as {}".format(fNum, foldSaveFilePath))
        pickle_save(
            foldSaveFilePath,
            {
                "foldRes": foldRes,
                "settings": settings,
                "CVFoldInds": CVFoldInds,
                "isTrain": isTrain,
                "isTest": isTest,
                "YShape": Y.shape if not isinstance(Y, (list, tuple)) else Y[0].shape,
                "ZShape": (
                    (Z.shape if not isinstance(Z, (list, tuple)) else Z[0].shape)
                    if Z is not None
                    else None
                ),
            },
        )

        if "removeFoldResults" in settings and settings["removeFoldResults"]:
            # Delete results for individual mi and nx values
            for mi, methodCode in enumerate(methodCodes):
                for nxi, nx in enumerate(nxVals):
                    foldSaveFilePathMNx = foldSaveFilePathMNxAll[mi][nxi]
                    if os.path.exists(foldSaveFilePathMNx):
                        logger.info(
                            "Deleting existing {} ({})".format(
                                foldSaveFilePathMNx,
                                bytes_to_string(os.path.getsize(foldSaveFilePathMNx)),
                            )
                        )
                        os.remove(foldSaveFilePathMNx)
    return foldRes, nxVals, True


def doCVedModelingForMethodFoldWithArgs(args):
    return doCVedModelingForMethodFold(**args)


def doCVedModelingForMethodFold(
    caseLogStr="",
    Y=None,
    Z=None,
    U=None,
    T=None,
    foldInds=None,
    nxVals=None,
    n1Vals=None,
    iCVResults=None,
    methodCode=None,
    settings=None,
    true_model=None,
    YType=None,
    ZType=None,
    UType=None,
    ZClasses=None,
    missing_marker=None,
    priorSIdAllNx=None,
    priorIdInfoAllNx=None,
    priorIdSettings=None,
    CVFoldInds=None,
    fNum=None,
    saveFilePath=None,
    savePathTail="",
):
    """doCVedModelingForMethodFold runs cross-validated model fitting for fold and one specific method

    Args:
        caseLogStr (str, optional): _description_. Defaults to ''.
        Y (_type_, optional): _description_. Defaults to None.
        Z (_type_, optional): _description_. Defaults to None.
        U (_type_, optional): _description_. Defaults to None.
        T (_type_, optional): _description_. Defaults to None.
        foldInds (_type_, optional): _description_. Defaults to None.
        nxVals (_type_, optional): _description_. Defaults to None.
        n1Vals (_type_, optional): _description_. Defaults to None.
        iCVResults (_type_, optional): _description_. Defaults to None.
        methodCode (_type_, optional): _description_. Defaults to None.
        settings (_type_, optional): _description_. Defaults to None.
        true_model (_type_, optional): _description_. Defaults to None.
        YType (_type_, optional): _description_. Defaults to None.
        ZType (_type_, optional): _description_. Defaults to None.
        UType (_type_, optional): _description_. Defaults to None.
        ZClasses (_type_, optional): _description_. Defaults to None.
        missing_marker (_type_, optional): _description_. Defaults to None.
        priorSIdAllNx (_type_, optional): _description_. Defaults to None.
        priorIdInfoAllNx (_type_, optional): _description_. Defaults to None.
        priorIdSettings (_type_, optional): _description_. Defaults to None.
        CVFoldInds (list of dict): train/test indices of the folds. Defaults to None.
        fNum (int): fold number to fit. Defaults to None.
        saveFilePath (_type_, optional): _description_. Defaults to None.
        savePathTail (str, optional): _description_. Defaults to ''.

    Returns:
        foldRes (dict): fold results
        nxVals (list): values of nx
    """
    if settings is None:
        settings = {}

    isTrain = foldInds["trainInds"]
    isTest = foldInds["testInds"]
    if "isCloseInd" in foldInds:  # the time points closest to bev measurement
        isCloseInd = foldInds["isCloseInd"]

    return_prescaling_copy = (
        "genDataFigs" in settings
        and settings["genDataFigs"]
        and saveFilePath is not None
    )
    outs = prepareTrainingAndTestData(
        Y,
        Z,
        foldInds,
        settings,
        U=U,
        YType=YType,
        ZType=ZType,
        missing_marker=missing_marker,
        return_prescaling_copy=return_prescaling_copy,
    )
    (
        YTrain,
        ZTrain,
        YTest,
        ZTest,
        UTrain,
        UTest,
        yMean,
        yStd,
        YTestSc,
        zMean,
        zStd,
        ZTestSc,
        uMean,
        uStd,
        UTestSc,
    ) = outs[:15]
    if return_prescaling_copy:
        YTrainMSBU, ZTrainMSBU, UTrainMSBU = outs[15:]

    isTrialBased = isinstance(Y, (list, tuple))

    if T is not None:
        TTrain = T[isTrain]
        TTest = T[isTest]
    else:
        TTrain, TTest = None, None

    if "avgPredSampleGroups" in settings:
        avgPredSampleGroupsTrain = [
            np.nonzero(np.isin(isTrain, s))[0] for s in settings["avgPredSampleGroups"]
        ]
    else:
        avgPredSampleGroupsTrain = None

    foldMethodRes = [{} for nxi in range(len(nxVals))]
    foldSaveFilePathM = [None for nxi in range(len(nxVals))]

    iCVResults = None
    useNxiInFNameTail = ("idUsingTrueNx" in settings and settings["idUsingTrueNx"]) or (
        "idUsingTrueNxz" in settings and settings["idUsingTrueNxz"]
    )

    WS = {}
    for nxi, nx in enumerate(nxVals):
        if (
            "nxValsToRun" in settings
            and len(settings["nxValsToRun"]) > 0
            and nx not in settings["nxValsToRun"]
        ):
            logger.info(
                "Skipping {caseLogStr} because nx={nx} is not in nxValsToRun="
                + "{}".format(settings["nxValsToRun"])
            )
            continue
        elif (
            "nxIndsToRun" in settings
            and len(settings["nxIndsToRun"]) > 0
            and -1 not in settings["nxIndsToRun"]
            and nxi not in settings["nxIndsToRun"]
        ):
            logger.info(
                f"Skipping {caseLogStr} because nx={nx} (nxi={nxi}) is not in nxIndsToRun="
                + "{}".format(settings["nxIndsToRun"])
            )
            continue

        foldResMethodNx = None

        if saveFilePath is not None:
            foldSaveFilePathMNx = (
                saveFilePath[:-2]
                + (f"_nx{nx}" if not useNxiInFNameTail else f"_nxi{nxi}")
                + ".p"
            )
            foldSaveFilePathM[nxi] = foldSaveFilePathMNx
            if os.path.exists(foldSaveFilePathMNx):
                logger.info(
                    'Loading results for nx={} from "{}"'.format(
                        nx, foldSaveFilePathMNx
                    )
                )
                try:
                    fD = pickle_load(foldSaveFilePathMNx)
                    # To do: check settings
                    foldResMethodNx = fD["foldRes"]
                except Exception as e:
                    logger.info(
                        'Could not load "{}". Deleting it... and moving on.'.format(
                            foldSaveFilePathMNx
                        )
                    )
                    os.remove(foldSaveFilePathMNx)

        if foldResMethodNx is None:
            if n1Vals is not None:
                n1 = n1Vals[np.min([nxi, len(n1Vals) - 1])]

            logger.info(f"Working on {caseLogStr}")

            already_done = False
            for nxip in range(nxi):
                if (
                    "nx" in foldMethodRes[nxip]
                    and "n1" in foldMethodRes[nxip]
                    and foldMethodRes[nxip]["nx"] == nx
                    and foldMethodRes[nxip]["n1"] == n1
                ):
                    logger.info(
                        "Nx={}, N1={} already fitted... will use the same results".format(
                            nx, n1
                        )
                    )
                    foldMethodRes[nxi] = copy.deepcopy(foldMethodRes[nxip])
                    already_done = True
                    break
            if already_done:
                continue

            if priorSIdAllNx is not None:  # Prior dynamic model fit
                priorSId = priorSIdAllNx[nxi]
                # if isinstance(priorSId, LatentDecoder):
                #     priorSId = priorSId.model_latent
                if hasattr(priorSId, "restoreModels"):
                    priorSId.restoreModels()
                priorIdInfo = priorIdInfoAllNx[nxi]
            else:
                priorSId = None
                priorIdInfo = None

            tic = time.perf_counter()
            sId, idInfo, YTrainF, ZTrainF, UTrainF, UTestF = doModelFit(
                YTrain,
                ZTrain=ZTrain,
                UTrain=UTrain,
                TTrain=TTrain,
                methodCode=methodCode,
                nx=nx,
                n1=n1,
                settings=settings,
                WS=WS,
                YTest=YTestSc,
                ZTest=ZTestSc,
                UTest=UTestSc,
                true_model=true_model,
                YType=YType,
                ZType=ZType,
                ZClasses=ZClasses,
                missing_marker=missing_marker,
                passZForSID="trainIndsY" not in foldInds,
                priorSId=priorSId,
                priorIdInfo=priorIdInfo,
                priorIdSettings=priorIdSettings,
                skip_zscoring=True,  # Backwards compatibility: zscoring is already handled in this function, no need to handle it in doModelFit
                saveFilePath=saveFilePath,
                savePathTail=savePathTail
                + (f"_nx{nx}" if not useNxiInFNameTail else f"_nxi{nxi}"),
                avg_sample_groups=avgPredSampleGroupsTrain,
            )

            sId = addScalingInfoToModelObject(
                sId,
                methodCode,
                ny=YTrainF.shape[1] if not isTrialBased else YTrainF[0].shape[1],
                nz=(
                    (ZTrainF.shape[1] if not isTrialBased else ZTrainF[0].shape[1])
                    if ZTrainF is not None
                    else None
                ),
                nu=(
                    (UTrainF.shape[1] if not isTrialBased else UTrainF[0].shape[1])
                    if UTrainF is not None
                    else None
                ),
                yMean=yMean,
                yStd=yStd,
                zMean=zMean if Z is not None else None,
                zStd=zStd if Z is not None else None,
                uMean=uMean if U is not None else None,
                uStd=uStd if U is not None else None,
            )

            toc = time.perf_counter()
            idTime = toc - tic

            if n1Vals is not None:
                idInfo["n1"] = n1
            if iCVResults is not None:
                idInfo["iCVResults"] = iCVResults

            perf, zPredTest, yPredTest, xPredTest = evalSysId(
                sId,
                YTest,
                ZTest,  # Original test data (any preprocessing will be done by the learned model)
                YTrainF,
                ZTrainF,
                UTest=UTestF,
                UTrain=UTrainF,
                trueSys=true_model,
                YType=YType,
                ZType=ZType,
                useXFilt=settings["useXFilt"],
                missing_marker=missing_marker,
            )
            perf["NTrain"] = len(isTrain)
            perf["NTest"] = len(isTest)
            perf["idTime"] = idTime

            if hasattr(sId, "discardModels"):
                sId.discardModels()  # Otherwise will be hard to save

            if isTrialBased:
                zPredSamples = len(zPredTest)
            elif zPredTest[0] is not None:
                zPredSamples = (
                    zPredTest.shape[0]
                    if not isinstance(zPredTest, (list, tuple))
                    else zPredTest[0].shape[0]
                )
            else:
                zPredSamples = None

            if zPredSamples is not None and zPredSamples != len(isTest):
                raise (
                    Exception(
                        "There is something wrong with the z prediction! It has the wrong shape ({}, but expected {} samples)!".format(
                            zPredTest.shape, len(isTest)
                        )
                    )
                )

            foldResMethodNx = {
                "idSys": sId,
                "idInfo": idInfo,
                "nx": nx,
                "n1": n1,
                "perf": perf,
                "settings": settings,
                "zPredTest": zPredTest,
                "yPredTest": yPredTest,
                "y1Train": YTrainF[0, :] if not isTrialBased else YTrainF[0][0, :],
                "z1Train": (
                    (ZTrainF[0, :] if not isTrialBased else ZTrainF[0][0, :])
                    if Z is not None
                    else ZTrainF
                ),
                "y1Test": YTest[0, :] if not isTrialBased else YTest[0][0, :],
                "z1Test": (
                    (ZTest[0, :] if not isTrialBased else ZTest[0][0, :])
                    if ZTest is not None
                    else None
                ),
            }

            if saveFilePath is not None:
                Path(foldSaveFilePathMNx).parent.mkdir(parents=True, exist_ok=True)
                pickle_save(
                    foldSaveFilePathMNx,
                    {"foldRes": foldResMethodNx, "settings": settings},
                )
                logger.info(
                    f'Saved results for {caseLogStr} in "{foldSaveFilePathMNx}"'
                )

        foldMethodRes[nxi] = foldResMethodNx

    return foldMethodRes, foldSaveFilePathM


class fitMethodSettings:
    def __init__(self, methodCode):
        self.methodCode = copy.copy(methodCode)
        self.discardU = "disU" in methodCode
        self.assumeUTest0 = "TU0" in methodCode
        self.hyperparamSearch = "HPSS" in methodCode
        self.gridSearchUpTo = "GSUT" in methodCode or self.hyperparamSearch

        # Default criteria, for backward compatibility
        self.iCV_criteria = "within_sem"
        self.iCV_peak_det_sem_multiplier = 1
        self.iCV_peak_det_ratio = 0.05
        within_ratio_vals, matches1 = extractFloatsFromRegex(
            r"w(\d+\.?\d+|\d+)pp", methodCode
        )
        if len(within_ratio_vals) > 0:
            self.iCV_criteria = "within_ratio"
            self.iCV_peak_det_ratio = within_ratio_vals[0] / 100
        within_sem_vals, matches2 = extractFloatsFromRegex(
            r"w(\d+\.?\d+|\d+)sem", methodCode
        )
        if len(within_sem_vals) > 0:
            self.iCV_criteria = "within_sem"
            self.iCV_peak_det_sem_multiplier = within_sem_vals[0]

        # Check if a horizon is specified in the method code
        methodCodeNoH = methodCode
        self.horizon = None
        if "_H" in methodCode and "SID" in methodCode:
            regex = r"_H(\d+)_(\d+)"  # H2_40  # Horizon
            matches = re.findall(regex, methodCode)
            if len(matches):
                iy, iz = matches[0]
                iy, iz = int(iy), int(iz)
                self.horizon = [iy, iz]
            else:
                regex = r"_H(\d+)"  # H2  # Horizon
                matches = re.findall(regex, methodCode)
                if len(matches):
                    iy = matches[0]
                    iy = int(iy)
                    self.horizon = [iy]
            if self.horizon is not None:
                methodCodeNoH = methodCode.replace(
                    "_H" + "_".join([str(ii) for ii in self.horizon]), ""
                )

        # Check if a value for n1 is specified in the method code
        self.nx = None
        if "_nx" in methodCodeNoH:
            regex = r"_nx(\d+)"  # nx16  # nx
            matches = re.findall(regex, methodCode)
            if len(matches):
                nx = matches[0]
                nx = int(nx)
            self.nx = nx
            methodCodeNoH = methodCodeNoH.replace("_nx{}".format(self.nx), "")

        self.n1 = None
        if "_n1_" in methodCodeNoH and "PSID" in methodCodeNoH:
            regex = r"_n1_(\d+)"  # n1_16  # n1
            matches = re.findall(regex, methodCode)
            if len(matches):
                n1 = matches[0]
                n1 = int(n1)
            self.n1 = n1
            methodCodeNoH = methodCodeNoH.replace("_n1_{}".format(self.n1), "")

        zShiftsF, matchF = extractIntsFromRegex(r"zShiftF(\d+)", methodCode)
        zShiftsP, matchP = extractIntsFromRegex(r"zShiftP(\d+)", methodCode)
        if len(zShiftsF) > 0:
            self.zShift = zShiftsF[0]
            methodCodeNoH = methodCodeNoH.replace(f"zShiftF{self.zShift}", "").replace(
                "__", "_"
            )
        elif len(zShiftsP) > 0:
            self.zShift = -zShiftsP[0]
            methodCodeNoH = methodCodeNoH.replace(f"zShiftP{-self.zShift}", "").replace(
                "__", "_"
            )
        else:
            self.zShift = 0

        self.y_count_process = "yCP" in methodCodeNoH
        self.z_count_process = "zCP" in methodCodeNoH

        self.predictWithXFilt = "xFilt" in methodCode
        self.predictWithXSmooth = "xSmth" in methodCode
        methodCodeNoH = methodCodeNoH.replace("xFilt", "").replace("__", "_")
        methodCodeNoH = methodCodeNoH.replace("xSmth", "").replace("__", "_")
        if methodCodeNoH[-1] == "_":
            methodCodeNoH = methodCodeNoH[:-1]

        self.zscore_per_dim = "_zspd" in methodCode
        methodCodeNoH = methodCodeNoH.replace("_zspd", "").replace("__", "_")
        self.zscore_inputs = "_zs" in methodCode
        methodCodeNoH = methodCodeNoH.replace("_zs", "").replace("__", "_")

        self.keepPostHocMapForICV = "kphm" in methodCode
        methodCodeNoH = methodCodeNoH.replace("kphm", "").replace("__", "_")

        # Check if number of epochs is specified in methodCode
        self.fit_epochs = None
        regex = r"epochs?(\d+)"  # epochs1000
        matches = re.finditer(regex, methodCodeNoH)
        for matchNum, match in enumerate(matches, start=1):
            self.fit_epochs = int(match.groups()[0])

        ensemble_cnt, out_matches = parseMethodCodeArgEnsemble(methodCodeNoH)
        if len(ensemble_cnt) > 0:
            self.ensemble_count = ensemble_cnt[0]
            methodCodeNoH = methodCodeNoH.replace(
                "_" + out_matches[0].group(), ""
            ).replace(out_matches[0].group(), "")
        else:
            self.ensemble_count = 1

        self.finetuneAgainAfterUnfreeze = "finetuneAAUnf" in methodCode
        methodCodeNoH = (
            methodCodeNoH.replace("redo_finetuneAAUnf", "")
            .replace("finetuneAAUnf", "")
            .replace("__", "_")
        )

        self.finetuneUnfreezed = "finetuneUnf" in methodCode
        methodCodeNoH = (
            methodCodeNoH.replace("redo_finetuneUnf", "")
            .replace("finetuneUnf", "")
            .replace("__", "_")
        )

        self.finetune = (
            "finetune" in methodCode
            or self.finetuneAgainAfterUnfreeze
            or self.finetuneUnfreezed
        )
        methodCodeNoH = (
            methodCodeNoH.replace("redo_finetune", "")
            .replace("finetune", "")
            .replace("__", "_")
        )

        # Check if we need to stack past N states/inputs for decoding
        self.stackPastNSamplesForPred = 0  # 0 means just use 1 sample, 1 means use one more (two latest samples), etc
        regex = r"StP(\d+)"  # StP4
        matches = re.finditer(regex, methodCodeNoH)
        for matchNum, match in enumerate(matches, start=1):
            stackPastNs = match.groups()
            self.stackPastNSamplesForPred = int(stackPastNs[0])
        methodCodeNoH = methodCodeNoH.replace(
            "_StP{}".format(self.stackPastNSamplesForPred), ""
        )

        methodCodeNoDisU = methodCodeNoH.replace("_disU", "")

        # Check if a posthoc mapping to behavior has to be fitted
        self.posthocMap = None
        self.methodCodeBase = methodCodeNoDisU
        if np.any(
            [
                pattern in methodCodeNoDisU
                for pattern in ["_OLS", "_RR", "_PRR", "_SVR", "_DR", "_KNNR"]
            ]
        ):
            self.posthoc_mapping = True
            for pattern in ["_OLS", "_RR", "_PRR", "_SVR", "_DR", "_KNNR"]:
                if pattern in methodCodeNoDisU:
                    posthoc_method = pattern[1:]
                    break
            methodCodeBU = copy.copy(methodCodeNoDisU)
            self.posthocMap = (
                posthoc_method + methodCodeBU.split("_" + posthoc_method)[-1]
            )
            self.methodCodeBase = "".join(methodCodeBU.split("_" + posthoc_method)[:-1])

        self.decUFromY = "_decUFromY" in methodCodeNoH
        self.fit_Cz_via_KF = False

        if "DPAD" in methodCode:
            self.method = "DPAD"
        elif (
            "RNNNDM" in methodCode or "RNNSID" in methodCode or "RNNNLSID" in methodCode
        ):  # "NL" name will be phased out
            self.method = "RNNNDM"
        elif np.any(
            [
                pattern in methodCodeNoDisU
                for pattern in [
                    "PSID",
                    "PSID_Cz",
                    "PSID_TU0",
                    "PSID_B0",
                    "PSID_K0",
                    "PSID_igU",
                    "PSID_igY",
                    "PSID_OLS",
                    "PSID_RR",
                    "PSID_SVR",
                    "PSID_DR",
                ]
            ]
        ):
            self.method = "SubspacePSID"
            self.fit_Cz_via_KF = "Cz" not in methodCode
        elif np.any(
            [
                pattern in methodCodeNoDisU
                for pattern in [
                    "SID",
                    "SID_disU",
                    "SID_TU0",
                    "SID_B0",
                    "SID_K0",
                    "SID_igU",
                    "SID_igY",
                    "SID_OLS",
                    "SID_RR",
                    "SID_SVR",
                    "SID_DR",
                    "NDM",
                    "NDM_disU",
                    "NDM_TU0",
                    "NDM_B0",
                    "NDM_K0",
                    "NDM_igU",
                    "NDM_igY",
                    "NDM_OLS",
                    "NDM_RR",
                    "NDM_SVR",
                    "NDM_DR",
                ]
            ]
        ):
            self.method = "SubspaceSID"
            self.fit_Cz_via_KF = "Cz" not in methodCode
        elif "Ideal" in methodCode:
            self.method = "Ideal"

        if self.method in ["SubspacePSID", "SubspaceSID"]:
            self.bw_on_residual = None
            self.updateKfKv = None
            self.updateQRS = None
            if "bwYZ" in methodCode:
                self.bw_on_residual = False
            if "bwZRes" in methodCode:
                self.bw_on_residual = True
            if "upKfKv" in methodCode:
                self.updateKfKv = True
            if "upQRS" in methodCode:
                self.updateQRS = True

        (
            self.steps_ahead,
            steps_ahead_loss_weights,
            matches,
        ) = parseMethodCodeArgStepsAhead(methodCode)

        self.setBTo0 = "_B0" in methodCode
        self.setKTo0 = "_K0" in methodCode
        self.ignoreUAfterFit = "_igU" in methodCode
        self.ignoreYAfterFit = "_igY" in methodCode

        self.supports_posthocMap = (
            self.method in ["DPAD", "RNNNDM", "SubspacePSID", "SubspaceSID", "KKF"]
            or "CEBRA" in self.method
        )

    def getICVMethods(self):
        if self.hyperparamSearch:
            # Construct list of hyperparams from
            subMethods, methodCodeBase = prepareHyperParameterSearchSpaceFromFileSpec(
                self.methodCode
            )
        elif self.method in ["DPAD", "RNNNDM"]:
            subMethods, methodCodeBase = prepareHyperParameterSearchSpaceFromMethodCode(
                self.methodCode
            )
        else:
            methodCodeBase = self.methodCode
            subMethods = [methodCodeBase]
            # raise(Exception(f'Automatic determination of innerCV parameters not implemented for "{self.method}"'))
        return subMethods, methodCodeBase


def doModelFit(
    YTrain,
    ZTrain=None,
    UTrain=None,
    TTrain=None,
    methodCode=None,
    nx=None,
    n1=None,
    settings={},
    WS={},
    YTest=None,
    ZTest=None,
    UTest=None,
    true_model=None,
    skip_zscoring=False,
    priorSId=None,
    priorIdInfo=None,
    priorIdSettings=None,
    YType=None,
    ZType=None,
    UType=None,
    ZClasses=None,
    missing_marker=None,
    passZForSID=False,
    saveFilePath="",
    savePathTail="",
    avg_sample_groups=None,
):
    """Fits a model with the given methodCode to the given training data

    Args:
        YTrain (np.ndarray): observation signal 1 (e.g. neural activity)
        ZTrain (np.ndarray, optional): observation sigal 2 (e.g. behavior). Defaults to None.
        UTrain (np.ndarray, optional): input signal (e.g. task instructions). Defaults to None.
        methodCode (str, optional): method code. Defaults to None.
        nx (int, optional): latent state dimension. Defaults to None.
        n1 (int, optional): latent state diemnsion for stage 1. Defaults to None.
        settings (dict, optional): Additional settings. Defaults to {}.
        WS (dict, optional): workspace struct from any prior runs on the exact
                            same data but with different nx. Defaults to {}.
        YTest (np.ndarray, optional): observation signal 1 (e.g. neural activity) test set. Defaults to None.
        ZTest (np.ndarray, optional): observation sigal 2 (e.g. behavior) test set. Defaults to None.
        UTest (np.ndarray, optional): input signal (e.g. task instructions) test set. Defaults to None.
        true_model (object, optional): true model if known in simulations. Defaults to None.
        priorSId (object, optional): results from a prior system identification to use as the dynamic part of the modeling
        priorIdInfo (dict, optional): idInfo results from the prior system identification to use as the dynamic part of the modeling
        priorIdSettings (dict, optional): settings from the prior system identification to use as the dynamic part of the modeling
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        UType (string, optional): data type of U. Defaults to None.
        ZClasses (np.array, optional): class values in Z. Defaults to None.
        missing_marker (number, optional): the marker value for missing data. Defaults to None.
        passZForSID (bool, optional): [description]. Defaults to False.
        saveFilePath (str, optional): [description]. Defaults to ''.
        avg_sample_groups ([type], optional): [description]. Defaults to None.

    Returns:
        sId (object): learned model
        YTrainF (np.ndarray): final used YTrain
        ZTrainF (np.ndarray): final used ZTrain
        UTrainF (np.ndarray): final used UTrain
        UTestT (np.ndarray): if discartion, etc happended to UTrain, same applied to UTest

    """
    YTrainF = YTrain
    ZTrainF = ZTrain
    UTrainF = UTrain
    YTestF = YTest  # Just for validation
    ZTestF = ZTest  # Just for validation
    UTestF = UTest  # Just for validation

    methodCodeBU = copy.copy(methodCode)
    methodSettings = fitMethodSettings(methodCode)

    if methodSettings.zscore_inputs and not skip_zscoring:
        zscoreY = YType == "cont"
        yMean, yStd = learnScaling(
            YTrainF,
            zscoreY,
            zscoreY,
            zscore_per_dim=methodSettings.zscore_per_dim,
            missing_marker=missing_marker,
        )
        YTrainF = applyGivenScaling(YTrainF, yMean, yStd, missing_marker=missing_marker)
        YTestF = applyGivenScaling(YTestF, yMean, yStd, missing_marker)

        zscoreZ = ZType == "cont"
        if ZTrainF is not None:
            zMean, zStd = learnScaling(
                ZTrainF,
                zscoreZ,
                zscoreZ,
                zscore_per_dim=methodSettings.zscore_per_dim,
                missing_marker=missing_marker,
            )
            ZTrainF = applyGivenScaling(ZTrainF, zMean, zStd, missing_marker)
            ZTestF = applyGivenScaling(ZTestF, zMean, zStd, missing_marker)

        zscoreU = UType == "cont"
        if UTrainF is not None:
            uMean, uStd = learnScaling(
                UTrainF,
                zscoreU,
                zscoreU,
                zscore_per_dim=methodSettings.zscore_per_dim,
                missing_marker=missing_marker,
            )
            UTrainF = applyGivenScaling(UTrainF, uMean, uStd, missing_marker)
            UTestF = applyGivenScaling(UTestF, uMean, uStd, missing_marker)
        else:
            UTestF = None

    if methodSettings.zShift != 0 and ZTrainF is not None:
        zShift = methodSettings.zShift
        # Shift z by this amount
        ZTrainF = (
            np.roll(ZTrainF, zShift, axis=0)
            if not isinstance(ZTrainF, list)
            else [np.roll(zt, zShift, axis=0) for zt in ZTrainF]
        )
        if ZTestF is not None:
            ZTestF = (
                np.roll(ZTestF, zShift, axis=0)
                if not isinstance(ZTestF, list)
                else [np.roll(zt, zShift, axis=0) for zt in ZTestF]
            )
        # To do: replace the incorrect edge with missing_marker
        logger.info(
            f"Shifted z by {methodSettings.zShift} (negative means shifted future into the past)"
        )

    YTestFT = transposeIf(YTestF)
    ZTestFT = transposeIf(ZTestF)

    UTrainFT = transposeIf(UTrainF)
    UTestFT = transposeIf(UTestF)

    if UTrain is not None:
        if methodSettings.discardU:
            logger.info("Discarding input U ({}) in modeling".format(UTrainF.shape))
            UTrainF, UTrainFT, UTestF, UTestFT = None, None, None, None
        if methodSettings.assumeUTest0:
            logger.info("Assuming UTest is 0 (regardless of true value)")
            UTestF = 0 * UTestF
            UTestFT = transposeIf(UTestF)

    idInfo = {}
    if priorIdInfo is not None:
        idInfo = priorIdInfo
    if methodSettings.gridSearchUpTo:
        if "iCVResults" not in idInfo and (
            methodSettings.posthocMap is None
            or methodSettings.keepPostHocMapForICV
            or (
                "keepPosthocMapForGSUT" in settings
                and settings["keepPosthocMapForGSUT"]
            )
        ):  # Grid search up to a certain setting
            methodCodesToCheck = []
            saveFilePaths = []
            priorMethodCode = methodCode.replace(
                f"_ensm{methodSettings.ensemble_count}", ""
            ).replace(f"ensm{methodSettings.ensemble_count}", "")
            if priorMethodCode != methodCode:
                methodCodesToCheck.append(priorMethodCode)
                saveFilePathThis = saveFilePath.replace(methodCode, priorMethodCode)
                saveDir = os.path.dirname(saveFilePathThis)
                saveFName = (
                    os.path.splitext(os.path.split(saveFilePathThis)[-1])[0]
                    + savePathTail
                    + "_iCV.p"
                )
                saveFilePaths.append(os.path.join(saveDir, saveFName))
            # In the current saveCode
            saveDir = os.path.dirname(saveFilePath)
            saveFName = (
                os.path.splitext(os.path.split(saveFilePath)[-1])[0]
                + savePathTail
                + "_iCV"
            )
            saveFilePaths.append(os.path.join(saveDir, saveFName))
            methodCodesToCheck.append(methodCode)
            for spi, (methodCodeThis, saveFilePathThis) in enumerate(
                zip(methodCodesToCheck, saveFilePaths)
            ):
                if spi < len(saveFilePaths) - 1 and not os.path.exists(
                    saveFilePathThis
                ):
                    continue
                subMethods, methodCodeBase = fitMethodSettings(
                    methodCodeThis
                ).getICVMethods()
                iCVRes = pickMethodHyperParamsWithInnerCV(
                    YTrain,
                    ZTrain=ZTrain,
                    UTrain=UTrain,
                    TTrain=TTrain,
                    methodCode=methodCode,
                    subMethods=subMethods,
                    nx=nx,
                    n1=n1,
                    settings=settings,
                    WS={},
                    YType=YType,
                    ZType=ZType,
                    ZClasses=ZClasses,
                    missing_marker=missing_marker,
                    criteria=methodSettings.iCV_criteria,
                    peak_det_sem_multiplier=methodSettings.iCV_peak_det_sem_multiplier,
                    peak_det_ratio=methodSettings.iCV_peak_det_ratio,
                    saveFilePath=saveFilePathThis,
                )
            idInfo["iCVResults"] = iCVRes
            methodCode = iCVRes["selectedMethodCode"]
        elif "iCVResults" in idInfo:
            iCVRes = idInfo["iCVResults"]
            if len(priorIdSettings["casesToRun"]) > 1:
                raise (Exception("Not supported"))
            priorMethodCode = priorIdSettings["casesToRun"][0]
            steps, steps_w, matches = parseMethodCodeArgStepsAhead(priorMethodCode)
            if steps is not None and steps != [1]:
                steps_pattern = "_" + "_".join([match.group() for match in matches])
            else:
                steps_pattern = ""
            priorMethodCodeBase = (
                priorMethodCode.replace(steps_pattern, "")
                .replace("redo_finetune", "")
                .replace("finetune", "")
                .replace("__", "_")
            )
            priorMethodCodeBase = (
                priorMethodCodeBase
                if priorMethodCodeBase[-1] != "_"
                else priorMethodCodeBase[:-1]
            )
            if priorMethodCodeBase in methodCode:
                methodCode = methodCode.replace(
                    priorMethodCodeBase, iCVRes["selectedMethodCode"]
                )
                logger.info(
                    'In a prior run with code "{}", "{}" was selected, so now we select the same and move forward with "{}"'.format(
                        priorMethodCode, iCVRes["selectedMethodCode"], methodCode
                    )
                )
            else:
                raise (Exception("Not supported"))
        else:
            raise (
                Exception(
                    "Not supported. Currently, posthoc readouts can only be fit for GSUT methods if a non-posthoc version of the method has been fitted before and loaded as prior method. This is because changing the posthoc method means the whole innerCV needs to run again."
                )
            )

        # if 'nxValsToRun' in settings and len(settings['nxValsToRun']) > 0 and 'ordersToSearch' in settings and settings['ordersToSearch'] != settings['nxValsToRun']:
        #     raise(Exception('Skipping running the main fitting because not all nxVals={} are done to be able to save it, only these were run: {}'.format(settings['ordersToSearch'], settings['nxValsToRun'])))

    methodSettings = fitMethodSettings(methodCode)

    if methodSettings.horizon is not None:
        horizonBU = settings["horizon"] if "horizon" in settings else None
        settings["horizon"] = methodSettings.horizon

    if methodSettings.n1 is not None:
        n1BU = n1
        n1 = methodSettings.n1

    if methodSettings.nx is not None:
        nxBU = nx
        nx = methodSettings.nx

    methodStr = "{} nx={}".format(methodCode, nx)
    logger.info('Working on method "{}"'.format(methodStr))

    if methodSettings.method == "DPAD" or methodSettings.method == "RNNNDM":
        methodCodeBU = copy.copy(methodCode)
        if methodSettings.posthocMap is not None:
            methodCode = methodSettings.methodCodeBase

        args = DPADModel.prepare_args(methodCode)
        if (
            "HL" in methodCode
            or "LSTM" in methodCode
            or (args["steps_ahead"] is not None and args["steps_ahead"] != [1])
        ):
            isFullyLinear = False
        else:
            isFullyLinear = True

        if methodSettings.method == "DPAD":
            n1This = n1
        else:
            n1This = 0

        if "tensorboard" in settings and settings["tensorboard"]:
            log_dir = os.path.join(saveFilePath + "_" + methodCode + "_log", f"nx{nx}")
        else:
            log_dir = ""

        if "log_dir" in settings and settings["log_dir"] is not None:
            log_dir = settings["log_dir"]

        if "fit_epochs" in settings:
            args["epochs"] = settings["fit_epochs"]
        elif methodSettings.fit_epochs is not None:
            args["epochs"] = methodSettings.fit_epochs

        # args['epochs'] = 5 #TEMP

        if methodSettings.predictWithXSmooth:
            args["bidirectional"] = True

        if not methodSettings.finetune:
            logger.info(f"DPAD.fit args: {args}")
            sId = DPADModel(log_dir=log_dir, missing_marker=missing_marker)
            sId.fit(
                transposeIf(YTrainF),
                transposeIf(ZTrainF),
                U=UTrainFT,
                nx=nx,
                n1=n1This,
                YType=YType,
                ZType=ZType,
                Y_validation=YTestFT,
                Z_validation=ZTestFT,
                U_validation=UTestFT,  # This is not generally useful
                true_model=true_model,
                **args,
            )
        else:
            sId = priorSId
            if not isinstance(sId, DPADModel):
                logger.info(
                    f"Converting prior fully linear model into a DPAD object for finetuning"
                )
                sId = DPADModel(log_dir=log_dir, missing_marker=missing_marker)
                sId.fit(
                    transposeIf(YTrainF),
                    transposeIf(ZTrainF),
                    U=UTrainFT,
                    nx=nx,
                    n1=n1This,
                    YType=YType,
                    ZType=ZType,
                    epochs=0,
                    init_model=priorSId,
                    use_existing_prep_models=True,
                )
                E = sId.initE
                EInv = np.linalg.inv(E)
                sBack = sId.getLSSM()
                sBack.applySimTransform(EInv)
                from ..tests.test_DPADModel import assert_params_are_close

                (
                    skipParams,
                    impossibleParams,
                    okParams,
                    errorParams,
                    errorParamsErr,
                ) = assert_params_are_close(priorSId, sBack)
                if len(errorParams) > 0:
                    raise (
                        Exception(
                            f"Error in initializing DPAD model with prior LSSM model exactly"
                        )
                    )

            trainableParam = None
            priorIdHasTrainedForcasting = (
                hasattr(priorSId, "steps_ahead")
                and np.all(
                    np.array(priorSId.steps_ahead) == np.array(args["steps_ahead"])
                )
                and hasattr(priorSId, "enable_forward_pred")
                and priorSId.enable_forward_pred
            )
            if args["steps_ahead"] is not None and args["steps_ahead"] != [1]:
                args["enable_forward_pred"] = True
                if not priorIdHasTrainedForcasting:
                    sId.set_multi_step_with_A_KC(
                        True
                    )  # [TEMP, will be reverted in sId.finetune] Enable multistep ahead with A_KC, not a separate A param
                sId.set_steps_ahead(args["steps_ahead"])
                if methodSettings.finetuneUnfreezed:
                    logger.info(f"finetuning all parameters")
                    trainableParams = {"base": True, "fw": True}
                elif methodSettings.finetuneAgainAfterUnfreeze:
                    if priorIdHasTrainedForcasting:
                        logger.info(
                            f"prior model has already finetuned forecasting, doing a second finetuning with all parameters"
                        )
                        trainableParams = {"base": True, "fw": True}
                    else:
                        logger.info(
                            f"prior model has no forecasting, doing an initial finetuning just with the new forecasting parameters (base params frozen)"
                        )
                        trainableParams = {"base": False, "fw": True}
                else:
                    trainableParams = {"base": False, "fw": True}
                    logger.info(
                        f"Prior model did not have forecasting, freezing the base model and just finetuning the new forecasting parameters (base params frozen)"
                    )
            sId.finetune(
                transposeIf(YTrain),
                transposeIf(ZTrain),
                U=transposeIf(UTrain),
                trainableParams=trainableParams,
                **args,
            )
            if (
                methodSettings.finetuneAgainAfterUnfreeze
                and not priorIdHasTrainedForcasting
            ):
                logger.info(
                    f"Finetuning with frozen base parameters is over, finetuning again with all parameters unfrozen"
                )
                trainableParams = {"base": True, "fw": True}
                sId.finetune(
                    transposeIf(YTrain),
                    transposeIf(ZTrain),
                    U=transposeIf(UTrain),
                    trainableParams=trainableParams,
                    **args,
                )

        if "model_save_path" in settings and settings["model_save_path"] is not None:
            sId.saveToFile(settings["model_save_path"], saveTFModels=True)
            sId.restoreModels()
        if (
            isFullyLinear
            and not args["skip_Cy"]
            and args["steps_ahead"] is None
            and not args["bidirectional"]
        ):
            sId = sId.getLSSM()  # Uncomment to just use KF for decoding
        # sId.discardModels()
        # sId.restoreModels()
        if methodSettings.method == "DPAD":
            sId.zDims = np.arange(1, 1 + min([n1, nx]))
        else:
            sId.zDims = []
        methodCode = copy.copy(methodCodeBU)
    else:
        raise (Exception('Method "{}" is not supported!'.format(methodCode)))

    if methodSettings.setBTo0 and hasattr(sId, "B"):
        sId.BBackUp = sId.B
        sId.changeParams({"B": sId.B * 0})

    if methodSettings.setKTo0 and hasattr(sId, "K"):
        sId.QBackUp = sId.Q
        del sId.Q  # To force predictor for for future use
        sId.KBackUp = sId.K
        sId.changeParams({"K": sId.K * 0})

    if methodSettings.ignoreUAfterFit:
        sId.changeParamsToDiscardU()
    if methodSettings.ignoreYAfterFit:
        sId.changeParamsToDiscardY()

    if isinstance(sId, LSSM):
        sId.predictWithXFilt = methodSettings.predictWithXFilt

    if isinstance(sId, LSSM):
        sId.predictWithXSmooth = methodSettings.predictWithXSmooth

    # sId.yPrepModel = yPrepModel

    if (
        hasattr(methodSettings, "steps_ahead")
        and methodSettings.steps_ahead is not None
    ):
        if isinstance(sId, (LSSM)):
            sId.steps_ahead = methodSettings.steps_ahead

    if methodSettings.zscore_inputs and not skip_zscoring:
        ny = (
            YTrainF[0].shape[1]
            if isinstance(YTrainF, list)
            else YTrainF.shape[1] if YTrainF is not None else 0
        )
        nz = (
            ZTrainF[0].shape[1]
            if isinstance(ZTrainF, list)
            else ZTrainF.shape[1] if ZTrainF is not None else 0
        )
        nu = (
            UTrainF[0].shape[1]
            if isinstance(UTrainF, list)
            else UTrainF.shape[1] if UTrainF is not None else 0
        )
        sId = addScalingInfoToModelObject(
            sId,
            methodCode,
            ny=ny,
            nz=nz,
            nu=nu,
            yMean=yMean,
            yStd=yStd,
            zMean=zMean if ZTrainF is not None else None,
            zStd=zStd if ZTrainF is not None else None,
            uMean=uMean if UTrainF is not None else None,
            uStd=uStd if UTrainF is not None else None,
        )

    return sId, idInfo, YTrainF, ZTrainF, UTrainF, UTestF


def prep_data(Y, Z, U, YType, ZType, UType, settings):
    """Prepared data for model fitting

    Args:
        Y (np.array): input data.
        Z (np.array): output data.
        U (np.array): external input data.
        YType (string): data type of Y.
        ZType (string): data type of Z.
        UType (string): data type of U.
        settings (dict): model fitting settings.

    Returns:
        Y (np.array): input data.
        Z (np.array): output data.
        U (np.array): external input data.
        YType (string): data type of Y.
        ZType (string): data type of Z.
        UType (string): data type of U.
        ZClasses (np.array): class values in Z.
    """
    if YType is None:  # Auto detect signal types
        if "YType" in settings and settings["YType"] is not None:
            YType = settings["YType"]
        else:
            YType = "cont"
    if ZType is None:  # Auto detect signal types
        if "ZType" in settings and settings["ZType"] is not None:
            ZType = settings["ZType"]
        elif "float" in str(Z.dtype):
            ZType = "cont"
        elif "int" in str(Z.dtype):
            ZType = "cat"
        else:
            raise (Exception("Not supported"))

    if ZType == "cat":
        ZClasses = np.unique(Z)
    else:
        ZClasses = None

    if "ZClasses" not in settings:
        settings["ZClasses"] = ZClasses
    ZClasses = settings["ZClasses"]

    if UType is None:  # Auto detect signal types
        if "UType" in settings and settings["UType"] is not None:
            UType = settings["UType"]
        else:
            UType = "cont"

    if isinstance(Y, (list, tuple)):
        if YType == "cont" and Y is not None and np.issubdtype(Y[0].dtype, np.floating):
            Y = [np.array(YThis, dtype=np.floating) for YThis in Y]
        if (
            ZType == "cont"
            and Z is not None
            and not np.issubdtype(Z[0].dtype, np.floating)
        ):
            Z = [np.array(ZThis, dtype=np.floating) for ZThis in Z]
        if (
            UType == "cont"
            and U is not None
            and not np.issubdtype(U[0].dtype, np.floating)
        ):
            U = [np.array(UThis, dtype=np.floating) for UThis in U]
    else:
        if (
            YType == "cont"
            and Y is not None
            and not np.issubdtype(Y.dtype, np.floating)
        ):
            Y = np.array(Y, dtype=np.floating)
        if (
            ZType == "cont"
            and Z is not None
            and not np.issubdtype(Z.dtype, np.floating)
        ):
            Z = np.array(Z, dtype=np.floating)
        if (
            UType == "cont"
            and U is not None
            and not np.issubdtype(U.dtype, np.floating)
        ):
            U = np.array(U, dtype=np.floating)
    return Y, Z, U, YType, ZType, UType, ZClasses


def prepareTrainingAndTestData(
    Y,
    Z,
    foldInds,
    settings,
    U=None,
    YType="cont",
    ZType="cont",
    missing_marker=None,
    return_prescaling_copy=False,
):
    """Prepares training and test data by applying any necessary zscoring

    Args:
        Y (np.array): input neural data
        Z (np.array): input behavior data
        foldInds (dict): structure with fold infomation
        settings (dict): modeling settings
        U (np.array, optional): input external input data. Defaults to None.
        YType (str, optional): type of neural data. Defaults to 'cont'.
        ZType (str, optional): type of behavior data. Defaults to 'cont'.
        missing_marker (number, optional): the marker for samples with missing data. Defaults to None.
        return_prescaling_copy (bool, optional): If True will return copy of data before 
            zscoring, may be useful for plotting. Defaults to False.

    Returns:
        YTrain, ZTrain, \
            YTest, ZTest, \
            UTrain, UTest, \
            yMean, yStd, YTestSc, \
            zMean, zStd, ZTestSc, \
            uMean, uStd, UTestSc: zscored training and test data and zcoring info
    """
    isTrain = foldInds["trainInds"]
    isTest = foldInds["testInds"]
    if "isCloseInd" in foldInds:  # the time points closest to bev measurement
        isCloseInd = foldInds["isCloseInd"]

    isTrialBased = isinstance(Y, (list, tuple))

    if "trainIndsY" in foldInds and foldInds["trainIndsY"] != isTrain:
        logger.info(
            "For Y data training samples are different: NTrain: {}".format(
                foldInds["trainIndsY"]
            )
        )
        YTrain = (
            Y[foldInds["trainIndsY"], :]
            if not isTrialBased
            else Y[foldInds["trainIndsY"]]
        )
    else:
        YTrain = sliceIf(Y, isTrain)
    ZTrain = sliceIf(Z, isTrain)
    YTest = sliceIf(Y, isTest)
    ZTest = sliceIf(Z, isTest)
    if U is not None:
        UTrain = sliceIf(U, isTrain)
        UTest = sliceIf(U, isTest)
        UTrainT = transposeIf(UTrain)
        UTestT = transposeIf(UTest)
    else:
        UTrain, UTrainT = None, None
        UTest, UTestT = None, None

    # g = sns.lineplot(T, Y[:, 1])
    # plt.show()

    if not isTrialBased:
        y_n_with_missing_marker = (
            np.sum(np.any(YTrain == missing_marker, axis=1))
            if missing_marker is not None
            else 0
        )
    else:
        y_n_with_missing_marker = np.sum(
            np.array(
                [
                    np.sum(np.any(YTrainThis == missing_marker, axis=1))
                    for YTrainThis in YTrain
                ]
            )
        )
    if y_n_with_missing_marker > 1:
        logger.info(
            'y data already has {} samples of "{}" => will be treated as missing data'.format(
                y_n_with_missing_marker, missing_marker
            )
        )
    if settings["yDiscardRatio"] > 0:
        if isTrialBased:
            raise (Exception("Trial based sample discartion is not supported yet"))
        YTrain = discardSamples(
            YTrain, missing_marker, settings["yDiscardRatio"], settings["yDiscardSeed"]
        )
        YTrainIsOk = ~np.any(
            np.logical_or(np.isnan(YTrain), YTrain == missing_marker), axis=1
        )
        logger.info(
            "{:g}% of y samples were discarded from the training data (only {} samples remain)".format(
                100 * settings["yDiscardRatio"], sum(YTrainIsOk)
            )
        )

    if ZTrain is not None:
        if not isTrialBased:
            z_n_with_missing_marker = (
                np.sum(np.any(ZTrain == missing_marker, axis=1))
                if missing_marker is not None
                else 0
            )
        else:
            z_n_with_missing_marker = (
                np.sum(
                    np.array(
                        [
                            np.sum(np.any(ZTrainThis == missing_marker, axis=1))
                            for ZTrainThis in ZTrain
                        ]
                    )
                )
                if missing_marker is not None
                else 0
            )
        if z_n_with_missing_marker > 1:
            logger.info(
                'z data already has {} samples of "{}" => will be treated as missing data'.format(
                    z_n_with_missing_marker, missing_marker
                )
            )
        if settings["zDiscardRatio"] > 0:
            if isTrialBased:
                raise (Exception("Trial based sample discartion is not supported yet"))
            ZTrain = discardSamples(
                ZTrain,
                missing_marker,
                settings["zDiscardRatio"],
                settings["zDiscardSeed"],
            )
            ZTrainIsOk = ~np.any(
                np.logical_or(np.isnan(ZTrain), ZTrain == missing_marker), axis=1
            )
            logger.info(
                "{:g}% of z samples were discarded from the training data (only {} samples remain)".format(
                    100 * settings["zDiscardRatio"], sum(ZTrainIsOk)
                )
            )

    if return_prescaling_copy:
        YTrainMSBU = YTrain.copy()
        ZTrainMSBU = ZTrain.copy() if ZTrain is not None else None
        UTrainMSBU = UTrain.copy() if UTrain is not None else None

    if (settings["removeYMean"] or settings["zScoreY"]) and YType == "cont":
        yMean, yStd = learnScaling(
            YTrain, settings["removeYMean"], settings["zScoreY"], missing_marker
        )
        YTrain = applyGivenScaling(YTrain, yMean, yStd, missing_marker)
        YTestSc = applyGivenScaling(YTest, yMean, yStd, missing_marker)
    else:
        ny = YTrain.shape[1] if not isinstance(YTrain, list) else YTrain[0].shape[1]
        yMean = np.zeros(ny)
        yStd = np.ones(ny)
        YTestSc = YTest

    if ZTrain is not None:
        if (settings["removeZMean"] or settings["zScoreZ"]) and ZType == "cont":
            zMean, zStd = learnScaling(
                ZTrain, settings["removeZMean"], settings["zScoreZ"], missing_marker
            )
            ZTrain = applyGivenScaling(ZTrain, zMean, zStd, missing_marker)
            ZTestSc = applyGivenScaling(ZTest, zMean, zStd, missing_marker)
        else:
            nz = ZTrain.shape[1] if not isinstance(ZTrain, list) else ZTrain[0].shape[1]
            zMean = np.zeros(nz)
            zStd = np.ones(nz)
            ZTestSc = ZTest

        if "keepTestInTrain" not in settings:
            settings["keepTestInTrain"] = False
        if settings["keepTestInTrain"]:
            ZTrain = ZTrain.copy()  # make sure things aren't modified permanently...
            if "isCloseInd" in foldInds and np.size(isCloseInd) > 0:
                trainCloseInds = np.nonzero(np.isin(isTrain, isCloseInd))[0]
                ZTrain[trainCloseInds, :] = missing_marker  # void out test data

                ZTestSc_tmp = np.ones(ZTestSc.shape) * missing_marker  # initialize
                testCloseInds = np.nonzero(np.isin(isTest, isCloseInd))[0]
                ZTestSc_tmp[testCloseInds, :] = ZTestSc[
                    testCloseInds, :
                ]  # copy test data.
                ZTestSc = ZTestSc_tmp

                ZTest_tmp = np.ones(ZTest.shape) * missing_marker  # initialize
                ZTest_tmp[testCloseInds, :] = ZTest[testCloseInds, :]  # copy test data.
                ZTest = ZTest_tmp
            else:
                sharedInds = np.nonzero(np.isin(isTrain, isTest))[0]
                ZTrain[sharedInds, :] = (
                    missing_marker  # void test Z with missing_marker
                )
    else:
        ZTestSc = ZTest
        zMean, zStd = None, None

    if U is not None:
        if settings["removeUMean"] or settings["zScoreU"]:
            uMean, uStd = learnScaling(
                UTrain, settings["removeUMean"], settings["zScoreU"], missing_marker
            )
            UTrain = applyGivenScaling(UTrain, uMean, uStd, missing_marker)
            UTestSc = applyGivenScaling(UTest, uMean, uStd, missing_marker)
            UTrainT = UTrain.T
        else:
            nu = UTrain.shape[1] if not isinstance(UTrain, list) else UTrain[0].shape[1]
            uMean = np.zeros(nu)
            uStd = np.ones(nu)
            UTestSc = UTest
    else:
        UTestSc = UTest
        uMean, uStd = None, None
    """
    for fldi, fld in enumerate(CVFoldInds):
        isInTest = np.isin(np.arange(T.size), fld['testInds'])
        isInTrain = np.isin(np.arange(T.size), fld['trainInds'])
        g = sns.lineplot(x=T, y=(1+fldi)*isInTest)
        g = sns.lineplot(x=T, y=-(1+fldi)*isInTrain)
    sns.scatterplot(x=T[Z[:, 0]!=missing_marker], y=Z[Z[:, 0]!=missing_marker, 0])
    plt.show()
    """
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.subplots(2, 1)
    ax[0].plot(isTrain, YTrain[:,0], label='Train')
    ax[0].plot(isTest, applyGivenScaling(YTest, yMean, yStd, missing_marker)[:,0], label='Test')
    ax[0].set_ylabel('Y1')
    ax[0].legend()
    ax[1].plot(isTrain, ZTrain[:,0], label='Train')
    ax[1].plot(isTest, applyGivenScaling(ZTest, zMean, zStd, missing_marker)[:,0], label='Test')
    # ax[1].plot(isTest, zPredTest, label='Pred')
    ax[1].set_ylabel('Z1')
    ax[1].legend()
    plt.show()
    """
    outs = (
        YTrain,
        ZTrain,
        YTest,
        ZTest,
        UTrain,
        UTest,
        yMean,
        yStd,
        YTestSc,
        zMean,
        zStd,
        ZTestSc,
        uMean,
        uStd,
        UTestSc,
    )
    if return_prescaling_copy:
        outs += YTrainMSBU, ZTrainMSBU, UTrainMSBU
    return outs


def addScalingInfoToModelObject(
    sId, methodCode, ny, nz, nu, yMean, yStd, zMean, zStd, uMean, uStd
):
    methodSettings = fitMethodSettings(methodCode)
    if methodSettings.methodCode == "Ideal":  # ‌ No need to scale data for true system
        if not hasattr(sId, "yMean"):
            sId.yMean = np.zeros(ny)
        if not hasattr(sId, "zMean"):
            sId.zMean = np.zeros(nz)
        if not hasattr(sId, "yStd"):
            sId.yStd = np.ones(ny)
        if not hasattr(sId, "zStd"):
            sId.zStd = np.ones(nz)
        if nu is not None and nu > 0:
            if not hasattr(sId, "uMean"):
                sId.uMean = np.zeros(nu)
            if not hasattr(sId, "uStd"):
                sId.uStd = np.ones(nu)
    else:
        if (
            (hasattr(sId, "yMean") and np.any(np.abs(sId.yMean - yMean) > 1e-9))
            or (hasattr(sId, "zMean") and np.any(np.abs(sId.zMean - zMean) > 1e-9))
            or (hasattr(sId, "yStd") and np.any(np.abs(sId.yStd - yStd) > 1e-9))
            or (hasattr(sId, "yStd") and np.any(np.abs(sId.zStd - zStd) > 1e-9))
            or (
                nu is not None
                and nu > 0
                and not methodSettings.assumeUTest0
                and (
                    (hasattr(sId, "uMean") and np.any(np.abs(sId.uMean - uMean) > 1e-9))
                    or (hasattr(sId, "uStd") and np.any(np.abs(sId.uStd - uStd) > 1e-9))
                )
            )
        ):
            raise (
                Exception(
                    "Unexpeted. Do not call addScalingInfoToModelObject more than one time with different mean/std values for the same sId, it is dangerous since previous scaling info can be overwritten and unexpected results may happen."
                )
            )
        sId.yMean = yMean
        sId.zMean = zMean
        sId.yStd = yStd
        sId.zStd = zStd
        if nu is not None and nu > 0 and not methodSettings.assumeUTest0:
            sId.uMean = uMean
            sId.uStd = uStd
    return sId


def addSelectedMethodInfoToIdInfo(idInfoAll, extraSettings):
    """idInfoAll and associated extraSettings are received and a 'selectedMethodCode' is added to each idInfo

    Args:
        idInfoAll (_type_): _description_
        extraSettings (_type_): _description_

    Returns:
        _type_: _description_
    """

    for ind, idInfo in np.ndenumerate(idInfoAll):
        if "iCVResults" in idInfo and "selectedInd" in idInfo["iCVResults"]:
            mCode = idInfo["iCVResults"]["settings"]["casesToRun"][
                idInfo["iCVResults"]["selectedInd"][0]
            ]
        else:
            mi = ind[0]
            mCode = extraSettings["casesToRun"][mi]
        idInfo["selectedMethodCode"] = mCode
    return idInfoAll


def loadFoldResults(saveFile, fNum, CVFoldInds, settings):
    """Loadings the model fitting results for one fold

    Args:
        saveFile (string): base path to the saved file.
        fNum (number): fold index.
        CVFoldInds (list of dict): list of training/test sets for folds.
        settings (dict): model fitting settings.

    Returns:
        foldRes (dict): loaded fold results
        foldSaveFilePath (string): name of the save file for the fold
    """

    foldRes, foldSaveFilePath = None, None
    if saveFile is not None:
        CVFolds = len(CVFoldInds)
        isTrain = CVFoldInds[fNum - 1]["trainInds"]
        isTest = CVFoldInds[fNum - 1]["testInds"]

        foldSaveFilePath = "{}.f{}_{}.p".format(saveFile, fNum, CVFolds)
        if os.path.exists(foldSaveFilePath):
            try:
                logger.info(
                    "Loading results for fold {} from {} ({}, modified: {})".format(
                        fNum,
                        foldSaveFilePath,
                        bytes_to_string(os.path.getsize(foldSaveFilePath)),
                        datetime.fromtimestamp(
                            os.path.getmtime(foldSaveFilePath)
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )
                fD = pickle_load(foldSaveFilePath)
                foldRes = fD["foldRes"]
                errVar = "CVFoldInds"
                if "CVFoldInds" in fD:
                    np.testing.assert_equal(fD["CVFoldInds"], CVFoldInds)
                else:
                    np.testing.assert_equal(foldRes[0][0]["CVFoldInds"], CVFoldInds)
                errVar = "isTrain"
                if "isTrain" in fD:
                    np.testing.assert_equal(fD["isTrain"], isTrain)
                errVar = "isTest"
                if "isTest" in fD:
                    np.testing.assert_equal(fD["isTest"], isTest)
                errVar = "ordersToSearch"
                if "settings" in fD:
                    np.testing.assert_equal(
                        fD["settings"]["ordersToSearch"], settings["ordersToSearch"]
                    )
                    for mi, mCode in enumerate(settings["casesToRun"]):
                        if len(foldRes[mi]) < len(settings["ordersToSearch"]) or np.any(
                            [fr == {} for fr in foldRes[mi]]
                        ):
                            errVar = "foldRes"
                            raise (Exception("Not all nx have results!"))
                        for nxi in range(len(foldRes[mi])):
                            zPredTest = foldRes[mi][nxi]["zPredTest"]
                            if isinstance(zPredTest, list) and isinstance(
                                zPredTest[0], list
                            ):  # Multi step, and trial-based
                                np.testing.assert_equal(len(zPredTest), len(isTest))
                            elif isinstance(zPredTest, list):
                                zPredTest = zPredTest[
                                    0
                                ]  # In case of multistep ahead, keep the first
                                np.testing.assert_equal(zPredTest.shape[0], len(isTest))
            except Exception as e:
                logger.warning(
                    "Loading failed or file results have inconsistencies with current settings... discarding. The error was:\n{}".format(
                        e
                    )
                )
                foldRes = None
    return foldRes, foldSaveFilePath


def loadFullCVedResults(saveFile, settings=None):
    if saveFile is not None and os.path.exists(saveFile) and os.path.isfile(saveFile):
        if saveFile.endswith(".p"):
            modTime = datetime.fromtimestamp(os.path.getmtime(saveFile))
            logger.info(
                "Loading results from {} ({}, modified: {})".format(
                    saveFile,
                    bytes_to_string(os.path.getsize(saveFile)),
                    modTime.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            if os.name == "nt":  # Running on windows
                PosixPathBU = pathlib.PosixPath
                pathlib.PosixPath = (
                    pathlib.WindowsPath
                )  # To help load torch models (e.g. CEBRA) fit on linux
            try:
                fD = pickle_load(saveFile)
            except Exception as e:
                logger.warning(e)
                if isinstance(
                    e, (EOFError, UnpicklingError)
                ):  # TO DO: before deleting, make sure error is related to pickle file and not something like out of memory
                    logger.info('Deleting corrupted file "{}"'.format(saveFile))
                    os.remove(saveFile)
                    return None, None, None, saveFile
                else:
                    raise (e)
            if os.name == "nt":  # Running on windows
                pathlib.PosixPath = PosixPathBU
            if (
                "idSysAll" in fD["CVFitRes"]
                and len(np.array(fD["CVFitRes"]["idSysAll"]).shape) != 3
            ):  # DOUBLE CHECK
                if (
                    len(np.array(fD["CVFitRes"]["idSysAll"]).shape) == 4
                    and np.array(fD["CVFitRes"]["idSysAll"]).shape[0] == 1
                ):
                    fD["CVFitRes"]["idSysAll"] = fD["CVFitRes"]["idSysAll"][0]
                    fD["CVFitRes"]["perfAll"] = fD["CVFitRes"]["perfAll"][
                        0
                    ]  # Dims: method, fold, nx
                    if "zPredAll" in fD["CVFitRes"]:
                        fD["CVFitRes"]["zPredAll"] = fD["CVFitRes"]["zPredAll"]
                else:
                    raise (Exception("Not expected!"))
            # Try reducing the memory footprint and the file size for future loads
            reduceCVedFitResultFile(saveFile, fD)
        elif saveFile.endswith(".mat"):
            logger.info(
                "Loading results from {} ({}, modified: {})".format(
                    saveFile,
                    bytes_to_string(os.path.getsize(saveFile)),
                    datetime.fromtimestamp(os.path.getmtime(saveFile)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
            )
            fD = mh.loadmat(saveFile)
        else:
            raise (ValueError("File extension not supported"))
    else:
        fD = None
    return fD


def loadAndParseFullCVedResults(
    saveFile, ZD, methodCodes, CVFolds, nxVals, CVFoldInds, settings, missing_marker
):
    fD = loadFullCVedResults(saveFile, settings)
    if fD is not None:
        CVFitRes = fD["CVFitRes"]
        CVFitRes, idSysAll, perfAll, CVFoldInds = parseLoadedCVResults(
            CVFitRes,
            ZD,
            methodCodes,
            CVFolds,
            nxVals,
            CVFoldInds,
            settings,
            missing_marker,
        )
        if perfAll is None:  # Results were corrupted
            logger.info('Deleting corruped file "{}"'.format(saveFile))
            os.remove(saveFile)
        else:
            badInds = []
            for i, (p, idSys) in enumerate(zip(perfAll.flatten(), idSysAll.flatten())):
                if (
                    "meanCC" in p
                    and np.isnan(p["meanCC"])
                    and "meanyCC" in p
                    and np.isnan(p["meanyCC"])
                ):
                    if (
                        isinstance(idSys, (DPADModel))
                        and hasattr(idSys, "XCov")
                        and np.any(np.isnan(idSys.XCov))
                    ):
                        logger.warning(
                            f"Some states have blownup in DPAD model #{i+1} of {perfAll.size} in this file"
                        )
                    badInds.append(i)
            if (
                len(badInds) > 0 and False
            ):  # Remove and False to delete any result file that has blown up performances
                logger.warning('Deleting file with bad model(s) "{}"'.format(saveFile))
                nametail = ""
                if os.path.exists(saveFile[:-2] + f"_BadSysBU{nametail}.p"):
                    if nametail == "":
                        nametail = 1
                    else:
                        nametail += 1
                os.rename(saveFile, saveFile[:-2] + f"_BadSysBU{nametail}.p")
                perfAll = None
    else:
        CVFitRes, idSysAll, perfAll = None, None, None
    return idSysAll, perfAll, CVFitRes


def loadAndParseFullCVedResultsIfExists(priorSaveFiles, priorLogStrs, *args, **kwargs):
    if priorLogStrs is None:
        priorLogStrs = ["" for p in priorSaveFiles]
    priorIdSysAll, priorPerfAll, priorCVFitRes = None, None, None
    for pi, (priorSaveFile, priorLogStr) in enumerate(
        zip(priorSaveFiles, priorLogStrs)
    ):
        if os.path.exists(priorSaveFile) or os.path.exists(priorSaveFile + ".p"):
            logger.warning(
                f"({pi+1}/{len(priorSaveFiles)}) Checked and dound results for a prior learned model here: {priorLogStr}"
            )
            (
                priorIdSysAll,
                priorPerfAll,
                priorCVFitRes,
                priorsaveFile,
            ) = loadAndParseFullCVedResults(priorSaveFile, *args, **kwargs)
            if priorIdSysAll is not None:
                logger.info('Loaded results from "{}"'.format(priorSaveFile))
                break
        else:
            logger.info(
                f"({pi+1}/{len(priorSaveFiles)}) Checked but did not find the following prior file: {priorSaveFile}"
            )
    return priorIdSysAll, priorCVFitRes


def parseLoadedCVResults(
    CVFitRes, Z, methodCodes, CVFolds, nxVals, CVFoldInds, settings, missing_marker
):
    """Parses CV results loaded from saved files

    Args:
        CVFitRes (dict): The CVFitRes field loaded from a saved file
        Z (np.ndarray): behavior data
        methodCodes (list of string): method codes
        CVFolds (int): number of CV folds
        nxVals (list of int): list of nx to fit
        CVFoldInds (list of dict): train/test indices of the folds
        settings (dict): dict with the run settings
        missing_marker (number): the marker value for missing data

    Returns:
        CVFitRes (dict): model fitting results
        idSysAll (object): learned models
        perfAll (dict): performance measures
        CVFoldInds (list of dicts): train/test indices for folds
    """
    idSysAll = (
        np.array(CVFitRes["idSysAll"])
        if "idSysAll" in CVFitRes
        else np.array(CVFitRes["idSysAllPerWin"])
    )
    perfAll = np.array(CVFitRes["perfAll"])  # Dims: method, fold, nx

    if type(CVFitRes["extraSettings"]["casesToRun"]) is np.ndarray:
        CVFitRes["extraSettings"]["casesToRun"] = list(
            CVFitRes["extraSettings"]["casesToRun"]
        )
    if type(CVFitRes["extraSettings"]["casesToRun"]) is not list:
        CVFitRes["extraSettings"]["casesToRun"] = [
            CVFitRes["extraSettings"]["casesToRun"]
        ]
    for i in range(len(CVFitRes["extraSettings"]["casesToRun"])):
        CVFitRes["extraSettings"]["casesToRun"][i] = CVFitRes["extraSettings"][
            "casesToRun"
        ][i].strip()
    if list(CVFitRes["extraSettings"]["casesToRun"]) != settings["casesToRun"]:
        logger.warning(
            "MethodCodes has changed from the following in the file \n{} to \n{}...".format(
                CVFitRes["extraSettings"]["casesToRun"], settings["casesToRun"]
            )
        )
        if (
            "takeMethodsFromSavedFile" in settings
            and settings["takeMethodsFromSavedFile"]
        ):
            settings["casesToRun"] = CVFitRes["extraSettings"]["casesToRun"]
            logger.warning("=>Taking methods to be those from the file...")
        else:
            logger.warning("=>Running again to overwrite the results...")
            perfAll = None
    if list(CVFitRes["extraSettings"]["ordersToSearch"]) != settings["ordersToSearch"]:
        logger.warning(
            "ordersToSearch has changed from the following in the file \n{} to \n{}...".format(
                CVFitRes["extraSettings"]["ordersToSearch"], settings["ordersToSearch"]
            )
        )
        if (
            "takeMethodsFromSavedFile" in settings
            and settings["takeMethodsFromSavedFile"]
        ):
            settings["ordersToSearch"] = CVFitRes["extraSettings"]["ordersToSearch"]
            logger.warning("=>Taking ordersToSearch to be those from the file...")
        else:
            logger.warning("=>Running again to overwrite the results...")
            perfAll = None
    if "CVFoldInds" in CVFitRes and np.any(
        [
            np.any(CVFitRes["CVFoldInds"][fi]["testInds"] != CVFoldInds[fi]["testInds"])
            for fi in range(len(CVFoldInds))
        ]
    ):
        logger.warning(
            "Fold settings (CVFoldInds) has changed has changed from the following in the file"
        )
        if (
            "takeMethodsFromSavedFile" in settings
            and settings["takeMethodsFromSavedFile"]
        ):
            CVFoldInds = CVFitRes["CVFoldInds"]
            logger.warning("=>Taking CVFoldInds to be those from the file...")
        else:
            logger.warning("=>Running again to overwrite the results...")
            perfAll = None
    elif (
        "CVFoldInds" not in CVFitRes
    ):  # For backward compatibility with older saved results
        logger.warning(
            "Warning: CVFoldInds is not saved... will assume that things are equal to their setting during the run of the saved results..."
        )
        CVFitRes["CVFoldInds"] = CVFoldInds
    if "idInfoAll" in CVFitRes:
        CVFitRes["idInfoAll"] = addSelectedMethodInfoToIdInfo(
            CVFitRes["idInfoAll"], CVFitRes["extraSettings"]
        )
    if (
        "missing_marker" in CVFitRes["extraSettings"]
        and missing_marker != CVFitRes["extraSettings"]["missing_marker"]
    ):
        logger.warning("WARNING: missing_marker has changed...")
        z_n_with_missing_marker = np.sum(np.any(Z == missing_marker, axis=1))
        if (
            z_n_with_missing_marker > 2
        ):  # 2 instead of 0 to ignore occasional coincidental cases of data actually being the missing_marker
            logger.warning(
                "Will discard saved data since there are some ({}) missing samples in the z data".format(
                    z_n_with_missing_marker
                )
            )
            perfAll = None
        else:
            logger.warning(
                "WARNING: assuming all will go well because no sample is missing in the data"
            )
    else:
        # Make sure dims are as follows:
        # idSysAll, perfAll: method x fold x nx
        # zPredAll: method x fold=1 x nx x time x dim
        # CVFitRes['z']['data']: time x dim
        if "z" in CVFitRes and len(CVFitRes["z"]["data"].shape) < 2:
            CVFitRes["z"]["data"] = np.expand_dims(
                np.array(CVFitRes["z"]["data"]), axis=1
            )
        if len(settings["casesToRun"]) == 1:
            if len(np.array(perfAll).shape) < 3:
                idSysAll = np.expand_dims(np.array(idSysAll), axis=0)
                perfAll = np.expand_dims(np.array(perfAll), axis=0)
            if (
                "perfAllFCat" in CVFitRes
                and len(np.array(CVFitRes["perfAllFCat"]).shape) < 3
            ):
                CVFitRes["perfAllFCat"] = np.expand_dims(
                    np.array(CVFitRes["perfAllFCat"]), axis=0
                )
            if "zPredAll" in CVFitRes and len(CVFitRes["zPredAll"].shape) < 4:
                CVFitRes["zPredAll"] = np.expand_dims(CVFitRes["zPredAll"], axis=0)
        if CVFitRes["extraSettings"]["CVFolds"] == 1:
            if len(np.array(perfAll).shape) < 3:
                idSysAll = np.expand_dims(np.array(idSysAll), axis=1)
                perfAll = np.expand_dims(np.array(perfAll), axis=1)
        if (
            "perfAllFCat" in CVFitRes
            and len(np.array(CVFitRes["perfAllFCat"]).shape) < 3
        ):
            CVFitRes["perfAllFCat"] = np.expand_dims(
                np.array(CVFitRes["perfAllFCat"]), axis=1
            )
        if (
            np.array(CVFitRes["extraSettings"]["ordersToSearch"]).size == 1
            and perfAll is not None
        ):
            if len(np.array(perfAll).shape) < 3:
                idSysAll = np.expand_dims(np.array(idSysAll), axis=2)
                perfAll = np.expand_dims(np.array(perfAll), axis=2)
            if (
                "perfAllFCat" in CVFitRes
                and len(np.array(CVFitRes["perfAllFCat"]).shape) < 3
            ):
                CVFitRes["perfAllFCat"] = np.expand_dims(
                    np.array(CVFitRes["perfAllFCat"]), axis=2
                )
            if "zPredAll" in CVFitRes and len(CVFitRes["zPredAll"].shape) < 4:
                CVFitRes["zPredAll"] = np.expand_dims(CVFitRes["zPredAll"], axis=1)
        if "zPredAll" in CVFitRes and len(CVFitRes["zPredAll"].shape) < 4:
            CVFitRes["zPredAll"] = np.expand_dims(CVFitRes["zPredAll"], axis=3)
        if "zPredAll" in CVFitRes and (
            "zPredAllUnit" not in CVFitRes or CVFitRes["zPredAllUnit"] == "normalized"
        ):
            if "notMissing" in CVFitRes:
                notMissing = CVFitRes["notMissing"]
            else:
                notMissing = ~np.any(Z == missing_marker, axis=1)
            notMissingInd = np.nonzero(notMissing)[0]
            allIsTest = []
            for fi in range(CVFolds):
                isTest = np.nonzero(
                    np.isin(notMissingInd, CVFitRes["CVFoldInds"][fi]["testInds"])
                )[0]
                isTestNew = ~np.isin(isTest, allIsTest)
                if np.any(~isTestNew):
                    logger.info(
                        "WARNINIG: There is {} samples of overlap between test data from fold {} and previous folds".format(
                            np.sum(~isTestNew), 1 + fi
                        )
                    )
                allIsTest.extend(isTest)
                for mi, methodCode in enumerate(methodCodes):
                    for nxi, nx in enumerate(nxVals):
                        idSys = idSysAll[mi][fi][nxi]
                        CVFitRes["zPredAll"][mi][0][nxi][isTest[isTestNew], :] = (
                            undoScaling(
                                idSys,
                                CVFitRes["zPredAll"][mi][0][nxi][isTest[isTestNew], :],
                                "zMean",
                                "zStd",
                                missing_marker=missing_marker,
                            )
                        )
            CVFitRes["zPredAllUnit"] = "z"
        if "chanSetId" in settings and "chanSetId" not in CVFitRes["extraSettings"]:
            CVFitRes["extraSettings"]["chanSetId"] = settings["chanSetId"]
            CVFitRes["extraSettings"]["kinSetId"] = settings["kinSetId"]
            CVFitRes["extraSettings"]["chanSet"] = settings["chanSet"]
            CVFitRes["extraSettings"]["kinSet"] = settings["kinSet"]

        for mi, methodCode in enumerate(settings["casesToRun"]):
            methodSettings = fitMethodSettings(methodCode)
            if (
                methodSettings.posthocMap is not None
                and methodSettings.supports_posthocMap
            ):
                idSys = idSysAll[mi, 0, 0]
                # if not isinstance(idSys, LatentDecoder):
                #     logger.info('Unexpected model for "{}", doesn''t have posthoc regression with "{}". Will discard and run again'.format(methodCode, methodSettings.posthocMap))
                #     return None, None, None, None # Return None to run again

    return CVFitRes, idSysAll, perfAll, CVFoldInds


def removeUnnecessaryInfoFromidInfoAll(idInfoAll, conservative=False):
    """Remove unnecessary fields from results to make saved files smaller

    Args:
        idInfoAll (list of dict): list of model learning info dictionaries
        conservative (bool, optional): if True, will keep fields that may be useful. Defaults to False.

    Returns:
        elimination_cnt (int): number of fields that were removed
    """
    elimination_cnt = 0
    for ind, idInfo in np.ndenumerate(idInfoAll):
        if "iCVResults" in idInfo and "CVFitRes" in idInfo["iCVResults"]:
            # Delete iCVResults to save memory
            CVFitRes = idInfo["iCVResults"]["CVFitRes"]
            if not conservative and CVFitRes is not None:
                idInfo["iCVResults"][
                    "CVFitRes"
                ] = None  # This will remove all info in CVFitRes
                elimination_cnt += 1
            elif isinstance(CVFitRes, dict):
                elim_something = False
                if "idSysAll" in CVFitRes:
                    for _, idSys in np.ndenumerate(CVFitRes["idSysAll"]):
                        if hasattr(idSys, "logs") and isinstance(idSys.logs, dict):
                            for f, v in idSys.logs.items():
                                if "epoch" in v and len(v["epoch"]) > 10:
                                    # delete item from dict
                                    v["epoch"] = v["epoch"][
                                        -1 :: -int(
                                            np.max([1, np.round(len(v["epoch"]) / 5)])
                                        )
                                    ]
                                    elim_something = True
                                if "history" in v:
                                    if "loss" in v["history"]:
                                        loss = v["history"]["loss"]
                                        if len(loss) > 10:
                                            v["history"] = {
                                                "loss": loss[
                                                    -1 :: -int(
                                                        np.max(
                                                            [1, np.round(len(loss) / 5)]
                                                        )
                                                    )
                                                ]
                                            }
                                            elim_something = True
                                    elif v["history"] is not {}:
                                        v["history"] = {}
                                        elim_something = True
                        rmFields = [
                            "YCov",
                            "YErrCov",
                            "YErrMean",
                            "ZCov",
                            "ZErrCov",
                            "ZErrMean",
                            "XCov",
                        ]
                        for fld in rmFields:
                            if hasattr(idSys, fld):
                                delattr(idSys, fld)
                                elim_something = True
                if "perfAll" in CVFitRes:
                    for _, perf in np.ndenumerate(CVFitRes["perfAll"]):
                        rmFields = [k for k in perf.keys() if "mean" not in k]
                        if len(rmFields) > 0:
                            [perf.pop(k) for k in rmFields]
                            elim_something = True
                if "perfAllFCat" in CVFitRes:
                    for _, perf in np.ndenumerate(CVFitRes["perfAllFCat"]):
                        rmFields = [k for k in perf.keys() if "mean" not in k]
                        if len(rmFields) > 0:
                            [perf.pop(k) for k in rmFields]
                            elim_something = True
                if (
                    "extraSettings" in CVFitRes
                    and "epochs" in CVFitRes["extraSettings"]
                ):
                    del CVFitRes["extraSettings"]["epochs"]
                    elim_something = True
                if elim_something:
                    elimination_cnt += 1
    return elimination_cnt


def reduceCVedFitResultFile(saveFile, fD=None):
    """Reduced the sice of CVed results for already saved results

    Args:
        saveFile (string): path to saved pickle file
        fD (dict, optional): saved results. Defaults to None.
    """
    if fD is None:
        logger.info(
            "Loading results from {} ({}, modified: {})".format(
                saveFile,
                bytes_to_string(os.path.getsize(saveFile)),
                datetime.fromtimestamp(os.path.getmtime(saveFile)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
        )
        fD = pickle_load(saveFile)
    # fDBU = copy.deepcopy(fD)
    logger.info("Checking whether result file size could be reduced for future loads")
    # Attempt to reduce file size by only removal minimal things from inner CV details (if any) from the file
    fieldNames = ["idInfoAll", "idInfoAllPerWin"]
    reductionCounter = 0
    for fieldname in fieldNames:
        if fieldname not in fD["CVFitRes"]:
            continue
        elimination_cnt = removeUnnecessaryInfoFromidInfoAll(
            fD["CVFitRes"][fieldname], conservative=True
        )
        if elimination_cnt > 0:
            if (
                removeUnnecessaryInfoFromidInfoAll(
                    fD["CVFitRes"][fieldname], conservative=True
                )
            ) > 0:
                raise (
                    Exception(
                        "Unexpected. Reduction is not expected to be applicable again after it is applied once! Check things!"
                    )
                )
            sFilePath = saveFile[:-2] + "_S.p"
            pickle_save(sFilePath, fD)
            if os.path.exists(saveFile):
                orig_size = os.path.getsize(saveFile)
                new_size = os.path.getsize(sFilePath)
                logger.info(
                    "Reduced the file size by {} ({} => {}), by removing some unnecessary inner CV results from {} models".format(
                        bytes_to_string(orig_size - new_size),
                        bytes_to_string(orig_size),
                        bytes_to_string(new_size),
                        elimination_cnt,
                    )
                )
                backupFilePath = saveFile[:-2] + "_BU.p"
                if os.path.exists(backupFilePath):
                    logger.info(
                        "Deleting existing {} ({})".format(
                            backupFilePath,
                            bytes_to_string(os.path.getsize(backupFilePath)),
                        )
                    )
                    os.remove(backupFilePath)
                os.rename(saveFile, backupFilePath)
            else:
                backupFilePath = None
            os.rename(sFilePath, saveFile)
            if backupFilePath is not None:
                logger.info(
                    "Renamed files:\nOld: ({}) {}\nNew: ({}) {}".format(
                        bytes_to_string(os.path.getsize(backupFilePath)),
                        os.path.basename(backupFilePath),
                        bytes_to_string(os.path.getsize(saveFile)),
                        os.path.basename(saveFile),
                    )
                )
            reductionCounter += 1

        # Attempt to reduce file size by removing inner CV details (if any) from the file
        elimination_cnt = removeUnnecessaryInfoFromidInfoAll(
            fD["CVFitRes"][fieldname], conservative=False
        )
        if elimination_cnt > 0:
            sFilePath = saveFile[:-2] + "_S.p"
            pickle_save(sFilePath, fD)
            orig_size = os.path.getsize(saveFile)
            new_size = os.path.getsize(sFilePath)
            logger.info(
                "Further reduced the file size by {} ({} => {}), by removing inner CV results from {} models".format(
                    bytes_to_string(orig_size - new_size),
                    bytes_to_string(orig_size),
                    bytes_to_string(new_size),
                    elimination_cnt,
                )
            )
            backupFilePath = saveFile[:-2] + "_withiCV.p"
            os.rename(saveFile, backupFilePath)
            os.rename(sFilePath, saveFile)
            logger.info(
                "Renamed files:\nOld: ({}) {}\nNew: ({}) {}".format(
                    bytes_to_string(os.path.getsize(backupFilePath)),
                    os.path.basename(backupFilePath),
                    bytes_to_string(os.path.getsize(saveFile)),
                    os.path.basename(saveFile),
                )
            )
            reductionCounter += 1

    if reductionCounter == 0:
        logger.info("Could not reduce file size")


def saveCVFitResults(CVFitRes, saveFile):
    """Saves cross-validated results in file

    Args:
        CVFitRes (dict): results of cross-validated modeling
        saveFile (string): path to save the file
    """
    saveDict = {"CVFitRes": CVFitRes}
    try:
        pickle_save(saveFile, saveDict)
        logger.info("Saving the results in {}".format(saveFile))
        reduceCVedFitResultFile(saveFile, saveDict)
    except Exception as e:
        logger.info(e)
        try:
            reduceCVedFitResultFile(saveFile, saveDict)
        except Exception as e:
            logger.info(e)
            logger.info(
                f"Pickle file didn't work, instead saving results as mat file in {saveFile}"
            )
            mh.savemat(saveFile, saveDict)


def removeFoldResultFiles(methodCodes, nxVals, CVFolds, foldSaveFiles):
    """Removes result files for individual folds

    Args:
        methodCodes (list of string): method codes
        nxVals (list): list of nx values
        CVFolds (number): number of CV folds
        foldSaveFiles (list of string): list of path to saved files
    """
    folds = np.arange(1, 1 + CVFolds)
    for fi, fNum in enumerate(folds):
        try:
            logger.info("Deleting {}".format(foldSaveFiles[fi]))
            os.remove(foldSaveFiles[fi])
            for mi, methodCode in enumerate(methodCodes):
                for nxi, nx in enumerate(nxVals):
                    if "GSUT" in methodCode:  # Grid search up to a certain setting
                        saveDir = os.path.dirname(foldSaveFiles[fi])
                        for savePathTail in [
                            "_mi{}_nxi{}".format(mi, nx),
                            "_mi{}_nx{}".format(mi, nx),
                            "_mi{}_nxi{}".format(mi, nxi),
                        ]:
                            saveFName = (
                                os.path.splitext(os.path.split(foldSaveFiles[fi])[-1])[
                                    0
                                ]
                                + savePathTail
                                + "_iCV"
                            )
                            saveFPath = os.path.join(saveDir, saveFName + ".p")
                            if os.path.exists(saveFPath):
                                logger.info("Deleting {}".format(saveFPath))
                                os.remove(saveFPath)
        except Exception as e:
            logger.info("Deletion failed with error: {}".format(e))


def prepareMethodSet(
    methodCodeBase, paramStr, paramSets, hidden_layer_cases, hidden_unit_cases
):
    """Prepare the list of methodCodes to consider in inner cross-validation

    Args:
        methodCodeBase (string): base of the methodCode
        paramStr (string): code for list of parameters
        paramSets (list of strings): list of model parameters
        hidden_layer_cases (np.array): list of hidden layer counts to consider
        hidden_unit_cases (np.array): list of hidden unit counts to consider

    Returns:
        subMethods (list of strings): list of method codes
    """
    subMethods = []
    if 0 in hidden_layer_cases:
        # The fully linear case:
        newMethodCode = (
            methodCodeBase.replace(paramStr, "").replace("_u_", "_").replace("__", "_")
        )
        subMethods.append(newMethodCode)
    paramSets = [pSet for pSet in paramSets if len(pSet) > 0]
    for pSet in paramSets:
        for HL in hidden_layer_cases:
            if HL < 1:
                continue
            for HU in hidden_unit_cases:
                paramStrThis = "".join(pSet) + f"{HL}HL{HU}U"
                newMethodCode = methodCodeBase.replace(paramStr, paramStrThis)
                subMethods.append(newMethodCode)
    for mi, mCode in enumerate(subMethods):
        args = DPADModel.prepare_args(mCode)
        if (
            "units" not in args["A_args"]
            or len(args["A_args"]["units"]) == 0
            or "units" not in args["K_args"]
            or len(args["K_args"]["units"]) == 0
        ):  # A and K are not both nonlinear
            subMethods[mi] = mCode.replace("_u", "_").replace("__", "_")
        while subMethods[mi][-1] == "_":  # Remove trailing _'s
            subMethods[mi] = subMethods[mi][:-1]
    return subMethods


def prepareHyperParameterSearchSpaceFromMethodCode(methodCode):
    """Prepares the nonlinearity options to be considered based on
    the description of the search space in the methodCode. When the methodCode
    contains the phrase "GSUT" (i.e., "Grid Search Up To"), a grid search over
    hyper parameters will be performed in a way that goes up to the specified
    values. For example, if 2 hidden layers are specified, 0, 1, and 2 hidden
    layers will be considered.

    Args:
        methodCode (string): overall method code. If it contains the phrase 'GSUT', then
            this method will return a list of method codes derived from the original
            method code that explore different nonlinearities.

    Returns:
        subMethods (list of string): list of method codes to consider in the hyperparameter search
        methodCodeBase (string): the base method code after removing parts about the hyperparameter search
    """
    # Check what settings the grid search should explore. Default is just the nonlinearity of the model parameters.
    gs_elems = ""
    gs_elems_str = ""
    regex = r"GSUTy?_([R|L|P|lr|wd]+)"  # L: lambda of regression, R: parameters that are regularized, P: nonlinearity of params, lr: learning rate
    matches = re.finditer(regex, methodCode)
    for matchNum, match in enumerate(matches, start=1):
        gs_elems = match.groups()[0]
        gs_elems_str = methodCode[match.span()[0] : match.span()[1]]
    grid_search_components = []
    if "P" in gs_elems:
        grid_search_components.append("param_nonlinearity")
    if "L" in gs_elems:
        grid_search_components.append("regularization_lambda")
    if "R" in gs_elems:
        grid_search_components.append("regularized_params")
    if "lr" in gs_elems:
        grid_search_components.append("learning_rate")
    if "wd" in gs_elems:
        grid_search_components.append("weight_decay")
    if gs_elems == "":  # By default, just grid search over parameter nonlinearities
        grid_search_components = ["param_nonlinearity"]

    mSettings = fitMethodSettings(methodCode)
    if not mSettings.gridSearchUpTo:
        subMethods = []
        methodCodeBase = methodCode
        return subMethods, methodCodeBase
    methodCodeBase = methodCode
    if gs_elems_str != "":
        methodCodeBase = methodCodeBase.replace(gs_elems_str, "").replace("__", "_")
    methodCodeBase = (
        methodCodeBase.replace("GSUTy", "").replace("GSUT", "").replace("__", "_")
    )
    ensemble_count = mSettings.ensemble_count
    methodCodeBase = (
        methodCodeBase.replace(f"_ensm{ensemble_count}", "")
        .replace(f"ensm{ensemble_count}", "")
        .replace("__", "_")
    )
    methodCodeBase = methodCodeBase.replace("kphm", "").replace("__", "_")

    out, out_matches = parseInnerCVFoldSettings(methodCodeBase)
    if len(out) > 0:
        iCVFolds = out[0]["folds"]
        iCVFoldsToConsider = out[0]["foldsToRun"]
        methodCodeBase = methodCodeBase.replace(out_matches[0].group(), "").replace(
            "__", "_"
        )

    if (
        "w1pp" in methodCodeBase
    ):  # Selection based on being within 1% of peak performance
        methodCodeBase = methodCodeBase.replace("w1pp", "").replace("__", "_")

    if (
        "kpp" in methodCodeBase
    ):  # Keeps this portion of the whole preprocessed data for all analyses
        _, _, re_match = parseMethodCodeArg_kpp(methodCodeBase)
        methodCodeBase = methodCodeBase.replace(re_match.group(), "").replace("__", "_")

    if "LR" in methodCode and "learning_rate" in grid_search_components:
        learning_rates, learning_rate_matches = extractPowRangesFromRegex(
            r"LR(\-?\d+)\^(\-?\d+);(\-?\d+);(\-?\d+)", methodCode, base_type=float
        )  # LR10^-5;1;-1
        learning_rates_pattern = "_" + "_".join(
            [match.group() for match in learning_rate_matches]
        )
        methodCodeBase = methodCodeBase.replace(learning_rates_pattern, "")

    if "WD" in methodCode and "weight_decay" in grid_search_components:
        weight_decays, weight_decay_matches = extractPowRangesFromRegex(
            r"LR(\-?\d+)\^(\-?\d+);(\-?\d+);(\-?\d+)", methodCode, base_type=float
        )  # WD10^-5;1;-1
        weight_decays_pattern = "_" + "_".join(
            [match.group() for match in weight_decay_matches]
        )
        methodCodeBase = methodCodeBase.replace(weight_decay_matches, "")

    subMethods = []
    params = ["A", "K", "Cz", "Dz", "A1", "K1", "Cz1", "Dz1", "A2", "K2", "Cz2", "Dz2"]
    if "skipCy" not in methodCode:
        params.extend(["Cy", "Dy", "Cy1", "Dy1", "Cy2", "Dy2"])
    if "HL" in methodCode and "param_nonlinearity" in grid_search_components:
        regex1 = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+);(\d+);(\d+)HL(\d+);(\d+);(\d+)U"  # 1;1;3HL64;2;128U
        regex2 = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+)HL(\d+);(\d+);(\d+)U"  # 1HL64;2;128U
        regex3 = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+);(\d+);(\d+)HL(\d+)U"  # 1;1;3HL128U
        regex4 = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+)HL(\d+)U"  # 1HL100U
        if len(re.findall(regex1, methodCode)) > 0:
            regex = regex1
        elif len(re.findall(regex2, methodCode)) > 0:
            regex = regex2
        elif len(re.findall(regex3, methodCode)) > 0:
            regex = regex3
        else:
            regex = regex4
        matches = re.finditer(regex, methodCode)
        for matchNum, match in enumerate(matches, start=1):
            if regex == regex1:
                (
                    var_names_str,
                    hidden_layers_min,
                    hidden_layers_step,
                    hidden_layers_max,
                    hidden_units_min,
                    hidden_units_multiple,
                    hidden_units_max,
                ) = match.groups()
            elif regex == regex2:
                (
                    var_names_str,
                    hidden_layers,
                    hidden_units_min,
                    hidden_units_multiple,
                    hidden_units_max,
                ) = match.groups()
                hidden_layers_min, hidden_layers_step, hidden_layers_max = (
                    0,
                    1,
                    hidden_layers,
                )
            elif regex == regex3:
                (
                    var_names_str,
                    hidden_layers_min,
                    hidden_layers_step,
                    hidden_layers_max,
                    hidden_units,
                ) = match.groups()
                hidden_units_min, hidden_units_multiple, hidden_units_max = (
                    64,
                    2,
                    hidden_units,
                )
            else:
                var_names_str, hidden_layers, hidden_units = match.groups()
                hidden_layers_min, hidden_layers_step, hidden_layers_max = (
                    0,
                    1,
                    hidden_layers,
                )
                hidden_units_min, hidden_units_multiple, hidden_units_max = (
                    64,
                    2,
                    hidden_units,
                )
            paramStr = methodCode[match.span()[0] : match.span()[1]]

            var_names = [p for p in params if p in var_names_str]
            if "A1" in var_names or "A2" in var_names:
                var_names.remove("A")
            if "K1" in var_names or "K2" in var_names:
                var_names.remove("K")
            if "Cy1" in var_names or "Cy2" in var_names:
                var_names.remove("Cy")
            if "Cz1" in var_names or "Cz2" in var_names:
                var_names.remove("Cz")

            hidden_layers_min = int(hidden_layers_min)
            hidden_layer_cases = np.arange(
                hidden_layers_min, 1 + int(hidden_layers_max), int(hidden_layers_step)
            )
            hidden_unit_base = int(hidden_units_multiple)
            hidden_unit_min_pow = np.floor(
                np.log(int(hidden_units_min)) / np.log(hidden_unit_base)
            )
            hidden_unit_max_pow = np.floor(
                np.log(int(hidden_units_max)) / np.log(hidden_unit_base)
            )
            hidden_unit_cases = np.array(
                hidden_unit_base
                ** np.arange(hidden_unit_min_pow, 1 + hidden_unit_max_pow),
                dtype=int,
            )

            paramSets = list(powerset(var_names))
            if "LSTM" in methodCode:
                methodCodeBaseWOLSTM = methodCodeBase.replace("LSTM", "").replace(
                    "__", "_"
                )
                subMethods2 = prepareMethodSet(
                    methodCodeBaseWOLSTM,
                    paramStr,
                    paramSets,
                    hidden_layer_cases,
                    hidden_unit_cases,
                )
                subMethods.extend(subMethods2)
                paramSets = [
                    ps for ps in paramSets if "A" not in ps
                ]  # Prepare for with LSTM. A is always nonlinear in LSTM, so no need to search over that

            subMethods1 = prepareMethodSet(
                methodCodeBase,
                paramStr,
                paramSets,
                hidden_layer_cases,
                hidden_unit_cases,
            )
            subMethods.extend(subMethods1)
    if len(subMethods) < 1:
        subMethods.append(methodCodeBase)

    if "LR" in methodCode and "learning_rate" in grid_search_components:
        subMethodsWithLR = []
        for lr in learning_rates:
            for sm in subMethods:
                newMethodsCode = sm
                newMethodsCode += f"_LR{lr:g}"
                subMethodsWithLR.append(newMethodsCode)
        subMethods = subMethodsWithLR

    if "WD" in methodCode and "weight_decay" in grid_search_components:
        subMethodsWithWD = []
        for wd in weight_decays:
            for sm in subMethods:
                newMethodsCode = sm
                newMethodsCode += f"_WD{wd:g}"
                subMethodsWithWD.append(newMethodsCode)
        subMethods = subMethodsWithWD

    if "RGL" in methodCode and (
        "regularized_params" in grid_search_components
        or "regularization_lambda" in grid_search_components
    ):
        # Regularization parameters
        regex = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)RGLB?(\d+)"  # _ARGL2_L1e5
        matches = re.finditer(regex, methodCode)
        for matchNum, match in enumerate(matches, start=1):
            var_names_str, norm_num = match.groups()
            paramStrSpan = match.span()
            paramStr = methodCode[paramStrSpan[0] : paramStrSpan[1]]
        if norm_num in ["1", "2"]:
            regularizer_name = "l{}".format(norm_num)
        else:
            raise (Exception("Unsupported method code: {}".format(methodCode)))
        lambdaVal = 0.01  # Default: 'l': 0.01

        var_names = [p for p in params if p in var_names_str]
        if "A1" in var_names or "A2" in var_names:
            var_names.remove("A")
        if "K1" in var_names or "K2" in var_names:
            var_names.remove("K")
        if "Cy1" in var_names or "Cy2" in var_names:
            var_names.remove("Cy")
        if "Cz1" in var_names or "Cz2" in var_names:
            var_names.remove("Cz")

        paramSets = list(powerset(var_names))
        paramSets = [pSet for pSet in paramSets if len(pSet) > 0]

        if "regularized_params" in grid_search_components:
            paramSetsSearch = paramSets
        else:
            paramSetsSearch = [()]

        # Whether the case with no regularization should be considered or not
        noRegCode = "_LNone"
        considerNoReg = (
            noRegCode in methodCode
        )  # Consider having no regularization as one of the cases

        # Regularization Lambda values
        methodCodeCpy = copy.copy(methodCode)
        lambdaVals, lambdaValStrs = extractValueRanges(methodCodeCpy, prefix="_L")
        for lvs in lambdaValStrs:
            methodCodeCpy = methodCodeCpy.replace(lvs, "")

        if len(lambdaVals) == 0:
            lambdaVals = [None]
            lambdaValStrs = ["_LNone"]

        if "regularization_lambda" in grid_search_components:
            lambdaValsSearch = lambdaVals
            lambdaValStrsSearch = lambdaValStrs
        else:
            lambdaValsSearch = [None]
            lambdaValStrsSearch = ["_LNone"]

        # Replace each methodCode with versions that have different regularization
        subMethodsWithReg = []

        for sm in subMethods:
            if considerNoReg:
                newSubMethod = copy.copy(sm)
                newSubMethod = newSubMethod.replace(noRegCode, "").replace(
                    "__", "_"
                )  # _LNone
                newSubMethod = newSubMethod.replace(paramStr, "").replace("__", "_")
                for lvStr2 in lambdaValStrs:  # Remove all lambdaValStrs
                    newSubMethod = newSubMethod.replace(lvStr2, "")
                subMethodsWithReg.append(newSubMethod)

            if len(paramSetsSearch) == 0:
                paramSetsSearch = [()]
            for pSet in paramSetsSearch:
                for lambdaVal, lambdaValStr in zip(
                    lambdaValsSearch, lambdaValStrsSearch
                ):
                    newSubMethod = copy.copy(sm)
                    newSubMethod = newSubMethod.replace(noRegCode, "").replace(
                        "__", "_"
                    )  # _LNone
                    if "regularized_params" in grid_search_components:
                        paramStrThis = "".join(pSet) + "RGL" + norm_num
                        newSubMethod = newSubMethod.replace(paramStr, paramStrThis)

                    if (
                        lambdaVal is not None
                        and "regularization_lambda" in grid_search_components
                    ):
                        newLVarStr = "_L{:.0e}".format(lambdaVal)
                        # Replace all other lambdaValStrs
                        for lvi, lvStr2 in enumerate(lambdaValStrs):
                            newSubMethod = newSubMethod.replace(
                                lvStr2, "" if lvi > 0 else newLVarStr
                            )
                    subMethodsWithReg.append(newSubMethod)
        subMethods = subMethodsWithReg
    return subMethods, methodCodeBase


def prepareHyperParameterSearchSpaceFromFileSpec(methodCode, settings=None):
    if settings is None:
        settings = {}

    # Get the search space code:
    outs, matches = extractStrsFromRegex(r"HPSS(_?[^_]*)", methodCode)

    if len(outs) == 0:
        raise (Exception(f"Code of HPSS (hyperparameter search space) not specified!"))

    HPSSMatchStr = matches[0].group()
    HPSSCode = outs[0]

    HPSSDir = os.path.join(Path(__file__).parent.parent.parent.parent, "HPSS")
    HPSSFilePath = os.path.join(HPSSDir, f"HPSS{HPSSCode}.yaml")

    if not os.path.exists(HPSSFilePath):
        raise (Exception(f"HPSS file not found: {HPSSFilePath}"))
    logger.info(
        f"Loading hyperparameter search space info from HPSS file: {HPSSFilePath}"
    )

    with open(HPSSFilePath) as stream:
        try:
            import yaml

            HPSS = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    all_cases = []
    if "parameters" in HPSS:
        params = []
        for param in HPSS["parameters"]:
            for key, val in param.items():
                this_param = {}
                this_param["name"] = key
                this_param["type"] = val["type"]
                this_param["dependent"] = "dependent" in val and val["dependent"]
                if "values" in val:
                    if this_param["dependent"]:
                        values = [
                            p["values"] for p in params if p["name"] == val["reference"]
                        ][0]
                        this_param["reference"] = val["reference"]
                    else:
                        if "format" in val:
                            if val["format"] == "csv":
                                values = val["values"].split(",")
                            elif val["format"] == "exprange":
                                values, value_matches = extractPowRangesFromRegex(
                                    r"(\-?\d+)\^(\-?\d+);(\-?\d+);(\-?\d+)",
                                    val["values"],
                                    base_type=float,
                                )  # 10^-5;1;-1
                            elif val["format"] == "linrange":
                                values, value_matches = extractLinearRangesFromRegex(
                                    r"([-+]?[\d]+\.?[\d]*);([-+]?[\d]+\.?[\d]*);([-+]?[\d]+\.?[\d]*)",
                                    val["values"],
                                    base_type=float,
                                )  # -5;1;-1
                            else:
                                raise (
                                    Exception(
                                        f"format "
                                        + val["format"]
                                        + f' not supported for parameter "{key}"'
                                    )
                                )
                        else:
                            raise (
                                Exception(
                                    f'"format" not specified for parameter "{key}"'
                                )
                            )
                    if this_param["type"] == "int":
                        values = [int(v) for v in values]
                    elif this_param["type"] == "float":
                        values = [
                            (
                                float(extractNumberFromRegex(v)[0][0])
                                if isinstance(v, str)
                                else v
                            )
                            for v in values
                        ]
                    this_param["values"] = values
                params.append(this_param)

        # Now expand to produce all combinations
        indep_params = [p for p in params if not p["dependent"]]
        indep_params_val_cases = [
            [{p["name"]: val} for val in p["values"]] for p in indep_params
        ]
        cases = [
            dict(ChainMap(*case)) for case in itertools.product(*indep_params_val_cases)
        ]

        dep_params = [p for p in params if p["dependent"]]
        for case in cases:
            for p in dep_params:
                case[p["name"]] = case[p["reference"]]
        all_cases.append(cases)

    if "architecture" in HPSS:
        param_options = {}
        paramCaseCodeGroups = []
        paramCaseSpecGroups = []
        for param in HPSS["architecture"]:
            for key, val in param.items():
                param_names = key.split(",")
                param_names_A = "".join(param_names)
                param_sets = []
                for set_size in range(1, 1 + len(param_names)):
                    param_sets.extend(
                        [
                            list(case)
                            for case in itertools.combinations(param_names, set_size)
                        ]
                    )
                paramCaseCodes = []
                paramCaseSpecs = []
                add_linear = False
                for option in val:
                    if isinstance(option, dict):
                        for name, settings in option.items():
                            if name.lower() == "mlp":  # Multilayer perceptron
                                layers = str(settings["hidden_layers"]).split(",")
                                units = str(settings["hidden_units"]).split(",")
                                NLCodes = [
                                    f"{case[0]}HL{case[1]}U"
                                    for case in itertools.product(*[layers, units])
                                ]
                                paramCaseCodes.extend(f"{NLCode}" for NLCode in NLCodes)
                                paramCaseSpecs.extend(
                                    {"MLP": NLCode} for NLCode in NLCodes
                                )
                    elif isinstance(option, str):
                        if option.lower() == "linear":
                            add_linear = True
                        elif option.lower() == "lstm":
                            paramCaseCodes.extend(["_LSTM"])
                            paramCaseSpecs.extend([{"LSTM": True}])
                    else:
                        raise (Exception("Unexpected!"))
                paramCaseCodesA = [
                    "".join(pSet) + pCaseCode
                    for pSet, pCaseCode in itertools.product(
                        *[param_sets, paramCaseCodes]
                    )
                ]
                paramCaseSpecsA = [
                    dict(ChainMap({"params": pSet}, pCaseCode))
                    for pSet, pCaseCode in itertools.product(
                        *[param_sets, paramCaseSpecs]
                    )
                ]
                if "" not in paramCaseCodesA and add_linear:
                    paramCaseCodesA.append("")
                    paramCaseSpecsA.append({})
                paramCaseSpecGroups.append(paramCaseSpecsA)

        def merge_specs(*dicts):
            merged = {}
            for d in dicts:
                for k, v in d.items():
                    if isinstance(d[k], list):
                        if k not in merged:
                            merged[k] = []
                        if isinstance(v, list):
                            merged[k].extend(
                                [elem for elem in v if elem not in merged[k]]
                            )
                    else:
                        merged[k] = v
            return merged

        NLSpecs = [
            merge_specs(*case) for case in itertools.product(*paramCaseSpecGroups)
        ]
        NLCodes2 = []
        for NLSpec in NLSpecs:
            code_parts = []
            if "LSTM" in NLSpec and NLSpec["LSTM"]:
                code_parts.append("LSTM")
                if "params" not in NLSpec or "A" not in NLSpec["params"]:
                    continue
            if "params" in NLSpec:
                p_str = "".join(NLSpec["params"])
                if ("unifiedAK" not in NLSpec or not NLSpec["unifiedAK"]) and (
                    "A" in NLSpec["params"] and "K" in NLSpec["params"]
                ):
                    p_str = "u" + p_str
                if (
                    "LSTM" in NLSpec
                    and NLSpec["LSTM"]
                    and (p_str == "A" or p_str == "uA")
                ):  # only A nonlinear and an LSTM
                    code_parts.append(p_str + "NonLin")
                elif "MLP" in NLSpec:
                    code_parts.append(p_str + NLSpec["MLP"])
                else:
                    code_parts.append(p_str)
            code = "_".join(code_parts)
            if code not in NLCodes2:
                NLCodes2.append(code)

        NLCodeCases = [{"archCode": code} for code in NLCodes2]
        all_cases.append(NLCodeCases)

    cases = [dict(ChainMap(*case)) for case in itertools.product(*all_cases)]

    methodCodeBase = copy.copy(methodCode)
    methodCodeBase = methodCodeBase.replace(HPSSMatchStr, "").replace("__", "_")

    out, out_matches = parseInnerCVFoldSettings(methodCodeBase)
    if len(out) > 0:
        iCVFolds = out[0]["folds"]
        iCVFoldsToConsider = out[0]["foldsToRun"]
        methodCodeBase = methodCodeBase.replace(out_matches[0].group(), "").replace(
            "__", "_"
        )

    subMethods = []
    for case in cases:
        thisMCode = methodCodeBase
        if "nx" in case:
            nx = case["nx"]
            thisMCode += f"_nx{nx}"
        if "n1" in case:
            n1 = case["n1"]
            thisMCode += f"_n1_{n1}"
        if "archCode" in case:
            archCode = case["archCode"]
            thisMCode += f"_{archCode}"
        if "learning_rate" in case:
            lr = case["learning_rate"]
            thisMCode += f"_LR{lr:g}"
        if "optimizer" in case:
            opt = case["optimizer"]
            thisMCode += f"_opt{opt}"
        if "weight_decay" in case:
            wd = case["weight_decay"]
            thisMCode += f"_WD{wd:g}"
        thisMCode = thisMCode.replace("__", "_")
        if thisMCode[-1] == "_":
            thisMCode = thisMCode[:-1]
        subMethods.append(thisMCode)

    return subMethods, methodCodeBase
