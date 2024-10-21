""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Some functions for plotting things"""

import copy
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
from scipy.stats import wilcoxon

from .evaluation import (
    computeMaskedStats,
    computePairwiseStatsTests,
    evalPrediction,
    evaluateDecoding,
    findPerformanceFrontier,
    getPerfVals,
    isAlmostAsGoodAsTheBest,
)
from .tools import getGapInds, shortenGaps

# matplotlib.use('Agg') # Uncomment to avoid the "failed to allocate bitmap" error. Comment to enable interactive GUI plots, but uncomment for final runs of scripts where many plots have to be generated and saved.

# import matplotlib
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
# import matplotlib.pyplot as plt
# pgf_with_latex = {
#     "text.usetex": True,            # use LaTeX to write all text
#     "pgf.rcfonts": False,           # Ignore Matplotlibrc
#     "pgf.preamble": [
#         r'\usepackage{color}'     # xcolor for colours
#     ]
# }
# matplotlib.rcParams.update(pgf_with_latex)


logger = logging.getLogger(__name__)


def rgb_to_hex(rgb):  # https://stackoverflow.com/q/29643352/2275605
    return "#%02x%02x%02x" % rgb


def plotPredictionScatter(
    trueVals,
    predVals,
    xValAxis=[0],
    xVals=[],
    legAxis=[],
    preMeanAxes=[],
    legNames=[],
    max_columns=4,
    xValField=None,
    yValFields=None,
    yValFieldsMeanElems=None,
    connect_sorted=False,
    show_hist_x=False,
    show_hist_y=False,
    addPerfMeasuresToTitle=[],
    addPerfMeasuresToLegend=[],
    missing_marker=None,
    plot45DegLine=False,
    plotLSLine=False,
    square=False,
    xLabel=None,
    yLabel=None,
    showSEM=True,
    showNoMeanLines=False,
    noMeanLabels=[],
    highlight_inds=None,
    highlight_styles=None,
    highlight_colormap=None,
    highlight_texts=[],
    highlights_connect=False,
    highlight_line_style=None,
    titleHead="",
    title="",
    titleTail="",
    x_keep_quantiles=None,
    XLim=None,
    YLim=None,
    XLimByQuantile=None,
    YLimByQuantile=None,
    max_out_dims_to_separate=0,
    xTakeLog10=False,
    yTakeLog10=False,
    skip_existing=False,
    saveFile=None,
    saveExtensions=None,
    return_fig=False,
    fig=None,
    figsize=None,
    colors=None,
    styles=None,
):
    """Plots a scatter plot of true vs predicted values

    Args:
        trueVals (_type_): _description_
        predVals (_type_): _description_
        xValAxis (list, optional): _description_. Defaults to [0].
        xVals (list, optional): _description_. Defaults to [].
        legAxis (list, optional): _description_. Defaults to [].
        preMeanAxes (list, optional): _description_. Defaults to [].
        legNames (list, optional): _description_. Defaults to [].
        xValField (_type_, optional): _description_. Defaults to None.
        yValFields (_type_, optional): _description_. Defaults to None.
        yValFieldsMeanElems (_type_, optional): _description_. Defaults to None.
        connect_sorted (bool, optional): _description_. Defaults to False.
        show_hist_x (bool, optional): _description_. Defaults to False.
        show_hist_y (bool, optional): _description_. Defaults to False.
        addPerfMeasuresToTitle (list, optional): _description_. Defaults to [].
        addPerfMeasuresToLegend (list, optional): _description_. Defaults to [].
        missing_marker (_type_, optional): _description_. Defaults to None.
        plot45DegLine (bool, optional): _description_. Defaults to False.
        plotLSLine (bool, optional): _description_. Defaults to False.
        square (bool, optional): _description_. Defaults to False.
        xLabel (_type_, optional): _description_. Defaults to None.
        yLabel (_type_, optional): _description_. Defaults to None.
        showSEM (bool, optional): _description_. Defaults to True.
        showNoMeanLines (bool, optional): _description_. Defaults to False.
        noMeanLabels (list, optional): _description_. Defaults to [].
        highlight_inds (_type_, optional): _description_. Defaults to None.
        highlight_styles (_type_, optional): _description_. Defaults to None.
        highlight_colormap (_type_, optional): _description_. Defaults to None.
        highlight_texts (list, optional): _description_. Defaults to [].
        highlights_connect (bool, optional): _description_. Defaults to False.
        highlight_line_style (_type_, optional): _description_. Defaults to None.
        titleHead (str, optional): _description_. Defaults to ''.
        title (str, optional): _description_. Defaults to ''.
        titleTail (str, optional): _description_. Defaults to ''.
        x_keep_quantiles (_type_, optional): _description_. Defaults to None.
        XLim (_type_, optional): _description_. Defaults to None.
        YLim (_type_, optional): _description_. Defaults to None.
        XLimByQuantile (_type_, optional): _description_. Defaults to None.
        YLimByQuantile (_type_, optional): _description_. Defaults to None.
        max_out_dims_to_separate (_type_, int): _description_. Defaults to 0.
        xTakeLog10 (bool, optional): _description_. Defaults to False.
        yTakeLog10 (bool, optional): _description_. Defaults to False.
        skip_existing (bool, optional): _description_. Defaults to False.
        saveFile (_type_, optional): _description_. Defaults to None.
        saveExtensions (_type_, optional): _description_. Defaults to None.
        return_fig (bool, optional): _description_. Defaults to False.
        fig (_type_, optional): _description_. Defaults to None.
        figsize (_type_, optional): _description_. Defaults to None.
        colors (_type_, optional): _description_. Defaults to None.
        styles (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if skip_existing and checkIfAllExtsAlreadyExist(saveFile, saveExtensions):
        logger.info("Skipping... figure already exists: " + saveFile)
        return

    if isinstance(trueVals, (list, tuple)):
        trueValsL = trueVals
    else:
        trueValsL = [trueVals]
    if isinstance(predVals, (list, tuple)):
        predValsL = predVals
    else:
        predValsL = [predVals]

    if not isinstance(titleHead, (list, tuple)):
        titleHead = [titleHead]
    if not isinstance(title, (list, tuple)):
        title = [title]
    if not isinstance(xLabel, (list, tuple)):
        xLabel = [xLabel]
    if not isinstance(yLabel, (list, tuple)):
        yLabel = [yLabel]

    out_dims = [tv.shape[0] for tv in trueValsL]
    if max(out_dims) <= max_out_dims_to_separate:
        trueValsLN = list(trueValsL[0])
        for li in range(1, len(trueValsL)):
            trueValsLN.extend(list(trueValsL[li]))
        predValsLN = list(predValsL[0])
        for li in range(1, len(predValsL)):
            predValsLN.extend(list(predValsL[li]))
        predValsLN = list(predValsL[0])
        for li in range(1, len(predValsL)):
            predValsLN.extend(list(predValsL[li]))
        titleN = [title[0]] * len(list(trueValsL[0]))
        for li in range(1, len(title)):
            titleN.extend([title[li]] * len(list(trueValsL[li])))
        if legNames is not None and len(legNames) == max(out_dims):
            for ti in range(len(titleN)):
                titleN[ti] += " " + legNames[ti % len(legNames)]
            legNames = []

        xLabelN = [xLabel[0]] * len(list(trueValsL[0]))
        for li in range(1, len(xLabel)):
            xLabelN.extend([xLabel[li % len(xLabel)]] * len(list(trueValsL[li])))
        xLabel = xLabelN

        yLabelN = [yLabel[0]] * len(list(trueValsL[0]))
        for li in range(1, len(yLabel)):
            yLabelN.extend([yLabel[li % len(yLabel)]] * len(list(trueValsL[li])))
        yLabel = yLabelN

        trueValsL = trueValsLN
        predValsL = predValsLN
        title = titleN

    if highlight_inds is None:
        highlight_inds = []

    num_plots = len(trueValsL)  # Plots in addition to legend
    if len(legNames):
        num_plots += 1
    columns = np.min([num_plots, max_columns])
    rows = int(np.ceil(num_plots / columns))

    if figsize is None:
        figsize = (2 * columns, 2 * rows)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        plt.figure(fig)
    # fig.subplots_adjust(hspace=0.25, wspace=0.25, bottom=0.2/figsize[1], top=1-0.4/figsize[1], left=0.3/figsize[0], right=1-0.3/figsize[0])
    fig.subplots_adjust(
        hspace=0.2, wspace=0.5, top=0.75, bottom=0.25, left=0.1, right=0.975
    )

    for thisXInd, thisX in enumerate(trueValsL):
        if thisX.ndim == 1:
            thisX = thisX[np.newaxis, :]
        thisY = predValsL[thisXInd]
        if thisY.ndim == 1:
            thisY = thisY[np.newaxis, :]

        ax = fig.add_subplot(rows, columns, 1 + thisXInd)
        prepAxesStyle(ax)

        defaultCols = plt.cm.get_cmap("tab10")(
            np.linspace(0, 1, np.max((thisY.shape[0], 10)))
        )[: thisY.shape[0]]
        defaultCols = [
            rgb_to_hex(tuple(np.round(defaultCols[ci][:3] * 255).astype(int)))
            for ci in range(defaultCols.shape[0])
        ]

        allX = []
        allY = []
        for ci in range(thisY.shape[0]):
            legName = None
            if len(legNames) > ci:
                if len(legNames[ci]) == 1:
                    legName = legNames[ci][0]
                elif isinstance(legNames[ci], str):
                    legName = legNames[ci]

            thisCol = defaultCols[np.mod(ci, len(defaultCols))]
            style = {"size": None, "marker": "o", "color": thisCol, "linestyle": "-"}
            if isinstance(styles, (tuple, list, np.ndarray)):
                if (
                    not isinstance(styles[ci % len(styles)], dict)
                    and len(styles[ci % len(styles)]) == 1
                ):
                    style.update(styles[ci % len(styles)][0])
                elif isinstance(styles[ci % len(styles)], dict):
                    style.update(styles[ci % len(styles)])
            elif type(styles) is dict:
                style.update(styles)

            x = thisX[ci % thisX.shape[0], :]
            y = thisY[ci, :]
            if x_keep_quantiles is not None:
                Qs = np.quantile(x, x_keep_quantiles)
                keep_ind = np.logical_and(x >= Qs[0], x <= Qs[1])
                x = x[keep_ind]
                y = y[keep_ind]
            allX.append(x)
            allY.append(y)
            if len(addPerfMeasuresToLegend) > 0:
                perfStrs = [
                    "{}:{:.3g}".format(
                        pm,
                        np.mean(
                            evalPrediction(
                                x[:, np.newaxis],
                                y[:, np.newaxis],
                                pm,
                                missing_marker=np.nan,
                            )
                        ),
                    )
                    for pm in addPerfMeasuresToLegend
                ]
                legName += "\n" + ", ".join(perfStrs)
            if connect_sorted:
                sInds = np.argsort(x)
                ax.plot(
                    x[sInds],
                    y[sInds],
                    label=legName,
                    c=style["color"],
                    linestyle=style["linestyle"],
                )
                if len(highlight_inds):
                    if highlight_styles is None and highlight_colormap is None:
                        ax.scatter(
                            x[highlight_inds],
                            y[highlight_inds],
                            s=style["size"],
                            c=style["color"],
                            marker=style["marker"],
                        )
                    else:
                        if highlight_colormap is not None:
                            cols = cm.get_cmap(plt.get_cmap(highlight_colormap))(
                                np.linspace(0, 1, len(highlight_inds))
                            )
                        for hi in highlight_inds:
                            if highlight_styles is not None:
                                s = highlight_styles[hi % len(highlight_styles)]
                            else:
                                s = style
                            styleC = copy.deepcopy(style)
                            styleC.update(s)
                            if highlight_colormap is not None:
                                styleC["color"] = rgb_to_hex(
                                    tuple(np.round(cols[hi, :3] * 255).astype(int))
                                )
                            ax.scatter(
                                x[highlight_inds[hi]],
                                y[highlight_inds[hi]],
                                s=styleC["size"],
                                c=styleC["color"],
                                marker=styleC["marker"],
                            )
                            if hi > 0 and highlights_connect:
                                if highlight_line_style is not None:
                                    styleC.update(highlight_line_style)
                                ax.plot(
                                    x[highlight_inds[(hi - 1) : (hi + 1)]],
                                    y[highlight_inds[(hi - 1) : (hi + 1)]],
                                    c=styleC["color"],
                                    linestyle=styleC["linestyle"],
                                    marker=styleC["marker"],
                                )
                            if (
                                highlight_texts is not None
                                and len(highlight_texts) > hi
                            ):
                                ax.text(
                                    x[highlight_inds[hi]],
                                    y[highlight_inds[hi]],
                                    highlight_texts[hi],
                                    ha="center",
                                    va="bottom",
                                    color="k",
                                )

            else:
                ax.scatter(
                    x,
                    y,
                    label=legName,
                    s=style["size"],
                    c=style["color"],
                    marker=style["marker"],
                )
            isOk = np.logical_and(~np.isnan(x), ~np.isnan(y))
            if missing_marker is not None:
                isOk = np.logical_and(isOk, x != missing_marker)
            if plotLSLine:
                try:
                    from PSID.PSID import projOrth

                    yHat, ba = projOrth(
                        y[isOk][np.newaxis, :],
                        np.concatenate(
                            (np.ones((1, x[isOk].size)), x[isOk][np.newaxis, :])
                        ),
                    )
                    # b, a = np.polynomial.polynomial.polyfit(x[isOk], y[isOk], 1)
                    # yHat = (b + a * x[isOk])[np.newaxis, :]
                    ax.plot(
                        x[isOk],
                        yHat.T,
                        linestyle=style["linestyle"],
                        color=style["color"],
                    )
                except Exception as e:
                    logger.info("Error in plotting LS line: {}".format(e))
            if plot45DegLine:
                if np.any(isOk):
                    v = np.array([np.min(x[isOk]), np.max(x[isOk])])
                else:
                    v = np.array([0, 1])
                ax.plot(v, v, linestyle="-", color="#C0C0C0", alpha=0.5)

        if XLimByQuantile is not None:
            for ci in range(thisY.shape[0]):
                x = thisX[ci % thisX.shape[0], :]
                thisQs = np.quantile(x, XLimByQuantile)
                if ci == 0:
                    XLim = thisQs
                else:
                    XLim = [np.min((XLim[0], thisQs[0])), np.max((XLim[1], thisQs[1]))]
        if YLimByQuantile:
            for ci in range(thisY.shape[0]):
                y = thisY[ci, :]
                thisQs = np.quantile(x, XLimByQuantile)
                if ci == 0:
                    XLim = thisQs
                else:
                    XLim = [np.min((XLim[0], thisQs[0])), np.max((XLim[1], thisQs[1]))]
        if XLim is None and YLim is None and plot45DegLine:
            v = np.concatenate((thisX.flatten(), thisY.flatten()))
            if np.any(~np.isnan(v)):
                XLim = np.array(
                    [
                        np.nanmin(
                            v,
                        ),
                        np.nanmax(v),
                    ]
                )
                XLim += 0.05 * np.diff(XLim) * np.array([-1, 1])
                YLim = XLim
            square = True
        if XLim is not None:
            ax.set_xlim(XLim)
        if YLim is not None:
            ax.set_ylim(YLim)
        for ci in range(thisY.shape[0]):
            thisCol = defaultCols[np.mod(ci, len(defaultCols))]
            x = allX[ci]
            y = allY[ci]
            curYLim = ax.get_ylim()
            curXLim = ax.get_xlim()
            if show_hist_x:
                hist, bin_edges = np.histogram(x[isOk], bins=100, density=True)
                bin_center = bin_edges[:-1] + np.diff(bin_edges) / 2
                ax.bar(
                    bin_center,
                    hist / np.max(hist) * 0.1 * np.diff(curYLim),
                    np.diff(bin_edges),
                    color=thisCol,
                    edgecolor="none",
                    alpha=0.5,
                    bottom=curYLim[0],
                )
                ax.set_ylim(curYLim)
            if show_hist_y:
                hist, bin_edges = np.histogram(y[isOk], bins=100, density=True)
                bin_center = bin_edges[:-1] + np.diff(bin_edges) / 2
                ax.barh(
                    bin_center,
                    hist / np.max(hist) * 0.1 * np.diff(curXLim),
                    np.diff(bin_edges),
                    color=thisCol,
                    edgecolor="none",
                    alpha=0.5,
                    left=curXLim[0],
                )
                ax.set_xlim(curXLim)
        xLabelThis = xLabel[thisXInd % len(xLabel)]
        if xLabelThis is None:
            xLabelThis = "True"
            if xTakeLog10:
                xLabelThis = "Log10 {}".format(xLabelThis)
        ax.set_xlabel(xLabelThis)
        yLabelThis = yLabel[thisXInd % len(yLabel)]
        if yLabelThis is None:
            yLabelThis = "Predicted"
            if yTakeLog10:
                yLabelThis = "Log10 {}".format(yLabelThis)
        ax.set_ylabel(yLabelThis)
        titleTailThis = copy.deepcopy(titleTail)
        if len(addPerfMeasuresToTitle) > 0:
            perfStrs = [
                "mean{}: {:.3g}".format(
                    pm,
                    np.mean(
                        evalPrediction(
                            np.array(allX).T,
                            np.array(allY).T,
                            pm,
                            missing_marker=np.nan,
                        )
                    ),
                )
                for pm in addPerfMeasuresToTitle
            ]
            titleTailThis += ", ".join(perfStrs)
        ax.set_title(
            "{}{}{}".format(
                titleHead[thisXInd % len(titleHead)],
                title[thisXInd % len(title)],
                titleTailThis,
            )
        )
        if square:
            ax.axis("square")
        if len(legNames) > 0 and (thisXInd == len(trueValsL) - 1):
            ax.legend(
                bbox_to_anchor=(1.04, 0.5),
                loc="center left",
                borderaxespad=0,
                fontsize="x-small",
            )
    if return_fig:
        return fig
    showOrSaveFig(fig, saveFile, saveExtensions)


def plotPerf(
    perfVals,
    perfVals2=None,
    xValAxis=[0],
    xVals=[],
    legAxis=[],
    preMeanAxes=[],
    legNames=[],
    textValsNoMean=[],
    peak_det_criteria="within_sem",
    peak_det_sem_multiplier=1,
    peak_det_ratio=0.05,
    xValField=None,
    yValFields=None,
    yValFieldsMeanElems=None,
    nxSelField=None,
    scatterInds=[],
    scatterLabels=[],
    write_peaks=False,
    write_peak_perfs=False,
    show_peaks_only=False,
    highlight_fronter=True,
    frontier_min_relative_diff=0,
    xLabel=None,
    yLabel=None,
    showSEM=True,
    showXValsSEM=False,
    showMeanLines=True,
    showNoMeanLines=False,
    noMeanLabels=[],
    lineMarkers=None,
    plot45DegLine=False,
    showStatTestLinePlot=False,
    showBox=False,
    showDots=False,
    writeMedian=False,
    showStatTest=False,
    statsVsFirst=False,
    showLetterGuides=True,
    titleHead="",
    title="",
    titleTail="",
    minYLim=None,
    YLim=None,
    XLim=None,
    XLimTight=False,
    x_ticks=None,
    x_tick_labels=None,
    x_scale=None,
    y_scale=None,
    x_scale_base=None,
    y_scale_base=None,
    xTakeLog2=False,
    yTakeLog2=False,
    xTakeLog10=False,
    yTakeLog10=False,
    saveFile=None,
    saveExtensions=None,
    figsize=(7, 4),
    colors=None,
    linestyles=None,
    scatterStyles=None,
    removeBadsFromAll=False,
    relaxNTrainCheck=False,
):
    """Plots a performance metric against another axis such as state dimension

    Args:
        perfVals (_type_): _description_
        perfVals2 (_type_, optional): _description_. Defaults to None.
        xValAxis (list, optional): _description_. Defaults to [0].
        xVals (list, optional): _description_. Defaults to [].
        legAxis (list, optional): _description_. Defaults to [].
        preMeanAxes (list, optional): _description_. Defaults to [].
        legNames (list, optional): _description_. Defaults to [].
        textValsNoMean (list, optional): _description_. Defaults to [].
        peak_det_criteria (str, optional): _description_. Defaults to 'within_sem'.
        peak_det_sem_multiplier (int, optional): _description_. Defaults to 1.
        peak_det_ratio (float, optional): _description_. Defaults to 0.05.
        xValField (_type_, optional): _description_. Defaults to None.
        yValFields (_type_, optional): _description_. Defaults to None.
        yValFieldsMeanElems (_type_, optional): _description_. Defaults to None.
        nxSelField (_type_, optional): _description_. Defaults to None.
        scatterInds (list, optional): _description_. Defaults to [].
        scatterLabels (list, optional): _description_. Defaults to [].
        write_peaks (bool, optional): _description_. Defaults to False.
        write_peak_perfs (bool, optional): _description_. Defaults to False.
        show_peaks_only (bool, optional): _description_. Defaults to False.
        highlight_fronter (bool, optional): _description_. Defaults to True.
        frontier_min_relative_diff (int, optional): _description_. Defaults to 0.
        xLabel (_type_, optional): _description_. Defaults to None.
        yLabel (_type_, optional): _description_. Defaults to None.
        showSEM (bool, optional): _description_. Defaults to True.
        showXValsSEM (bool, optional): _description_. Defaults to False.
        showMeanLines (bool, optional): _description_. Defaults to True.
        showNoMeanLines (bool, optional): _description_. Defaults to False.
        noMeanLabels (list, optional): _description_. Defaults to [].
        lineMarkers (_type_, optional): _description_. Defaults to None.
        plot45DegLine (bool, optional): _description_. Defaults to False.
        showStatTestLinePlot (bool, optional): _description_. Defaults to False.
        showBox (bool, optional): _description_. Defaults to False.
        showDots (bool, optional): _description_. Defaults to False.
        writeMedian (bool, optional): _description_. Defaults to False.
        showStatTest (bool, optional): _description_. Defaults to False.
        statsVsFirst (bool, optional): _description_. Defaults to False.
        showLetterGuides (bool, optional): _description_. Defaults to False.
        titleHead (str, optional): _description_. Defaults to ''.
        title (str, optional): _description_. Defaults to ''.
        titleTail (str, optional): _description_. Defaults to ''.
        minYLim (_type_, optional): _description_. Defaults to None.
        YLim (_type_, optional): _description_. Defaults to None.
        XLim (_type_, optional): _description_. Defaults to None.
        XLimTight (bool, optional): _description_. Defaults to False.
        x_ticks (_type_, optional): _description_. Defaults to None.
        x_tick_labels (_type_, optional): _description_. Defaults to None.
        x_scale (_type_, optional): _description_. Defaults to None.
        y_scale (_type_, optional): _description_. Defaults to None.
        x_scale_base (_type_, optional): _description_. Defaults to None.
        y_scale_base (_type_, optional): _description_. Defaults to None.
        xTakeLog2 (bool, optional): _description_. Defaults to False.
        yTakeLog2 (bool, optional): _description_. Defaults to False.
        xTakeLog10 (bool, optional): _description_. Defaults to False.
        yTakeLog10 (bool, optional): _description_. Defaults to False.
        saveFile (_type_, optional): _description_. Defaults to None.
        saveExtensions (_type_, optional): _description_. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (7, 4).
        colors (_type_, optional): _description_. Defaults to None.
        linestyles (_type_, optional): _description_. Defaults to None.
        scatterStyles (_type_, optional): _description_. Defaults to None.
        relaxNTrainCheck (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if len(yValFields) == 0:
        return
    if not showStatTestLinePlot:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(
            hspace=0.3, wspace=0.3, top=0.9, bottom=0.2, left=0.2, right=0.95
        )
        ax = fig.add_subplot(1, 2, 1)
        prepAxesStyle(ax)
    else:
        hr = 1.9
        fig = plt.figure(
            figsize=figsize * np.array([1, hr])
        )  # Make space for the stats plot
        fig.subplots_adjust(
            hspace=0.3 / hr,
            wspace=0.3,
            top=1 - 0.1 / hr,
            bottom=0.2 / hr + (hr - 1) / hr,
            left=0.2,
            right=0.95,
        )
        ax = fig.add_subplot(1, 2, 1)
        prepAxesStyle(ax)
        gs = fig.add_gridspec(
            1,
            2,
            hspace=0.3,
            wspace=0.3,
            top=1 - 1 / hr,
            bottom=0.1 / hr,
            left=0.2,
            right=0.95,
        )
        ax2 = fig.add_subplot(gs[0, 0], sharex=ax)
        prepAxesStyle(ax2)
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    if linestyles is None:
        linestyles = ["-"]
    lineMarkersBU = copy.copy(lineMarkers)
    if lineMarkers is None:
        lineMarkers = [""]
    if scatterStyles is None:
        scatterStyles = [
            {"marker": "o", "facecolors": "none"},
            {"marker": "+"},
            {"marker": "x"},
            {"marker": "*", "facecolors": "none"},
            {"marker": "1"},
            {"marker": "2"},
            {"marker": "3"},
            {"marker": "4"},
        ]
    cInd = -1

    source_data = []

    allRes, allRes2 = [], []
    for yFieldInd, yField in enumerate(yValFields):
        if yValFieldsMeanElems is None:
            meanElems = None
        else:
            meanElems = yValFieldsMeanElems[yFieldInd % len(yValFieldsMeanElems)]

        res = getPerfVals(
            perfVals,
            yField,
            xValField=xValField,
            nxSelField=nxSelField,
            xValAxis=xValAxis,
            xVals=xVals,
            meanElems=meanElems,
            groupAxis=legAxis,
            preMeanAxes=preMeanAxes,
            peak_det_criteria=peak_det_criteria,
            peak_det_sem_multiplier=peak_det_sem_multiplier,
            peak_det_ratio=peak_det_ratio,
        )
        allRes.append(res)

        if perfVals2 is not None:
            res2 = getPerfVals(
                perfVals2,
                yField,
                xValField=xValField,
                nxSelField=nxSelField,
                xValAxis=xValAxis,
                xVals=xVals,
                meanElems=meanElems,
                groupAxis=legAxis,
                preMeanAxes=preMeanAxes,
                peak_det_criteria=peak_det_criteria,
                peak_det_sem_multiplier=peak_det_sem_multiplier,
                peak_det_ratio=peak_det_ratio,
            )
            allRes2.append(res2)
            yValsAll2 = res2["yValsAll"]
            if show_peaks_only:
                yValsAll2 = res2["peakPerfYValAll"]

        def getBarPos(xValsThis, groupRelWith=0.5, li=0):
            if len(xValsThis) > 1:
                allbars_dx = np.min(
                    np.abs(np.diff(xValsThis))
                )  # Distance between x-points
            else:
                allbars_dx = 1.0
            groupRelWith = 0.5
            bar_dx = groupRelWith * allbars_dx / len(res["xValsAll"])
            x0 = xValsThis + np.array(
                -groupRelWith * allbars_dx / 2 + li * bar_dx + bar_dx / 2
            )
            return x0, bar_dx

        xValsAll = res["xValsAll"]
        yValsAll = res["yValsAll"]
        if show_peaks_only:
            xValsAll = [v[np.newaxis, :] for v in res["peakPerfXValAll"]]
            yValsAll = [v[np.newaxis, :] for v in res["peakPerfYValAll"]]

        if removeBadsFromAll:
            badInds = []
            for mi in range(len(res["yValsAll"])):
                yValsM = res["yValsAll"][mi]
                isBad = np.logical_or(np.isnan(yValsM), np.isinf(yValsM))
                badInds.extend(list(np.where(isBad)[1]))
            badInds = list(np.unique(badInds))
            xValsAllBU = copy.copy(xValsAll)
            yValsAllBU = copy.copy(yValsAll)
            for mi in range(len(res["yValsAll"])):
                isBad = np.isin(np.arange(yValsAll[mi].size), badInds)
                xValsAll[mi] = xValsAll[mi][:, ~isBad]
                yValsAll[mi] = yValsAll[mi][:, ~isBad]
            title += f"[{yValsAll[0].size}/{yValsAllBU[0].size} ok]"

        finalXVals = [None for li in range(len(xValsAll))]
        finalLabels = ["" for li in range(len(xValsAll))]
        for li in range(len(xValsAll)):
            cInd += 1
            yVals = yValsAll[li]
            if yTakeLog2:
                yVals = np.log2(yVals)
            if yTakeLog10:
                yVals = np.log10(yVals)
            yValsBad = np.logical_or(np.isnan(yVals), np.isinf(yVals))
            if np.all(yValsBad):
                logger.warning(f"All {sum(yValsBad)} perfs were nan")
                continue
            yValsMean, yValsMedian, yValsStd, yValsSEM = computeMaskedStats(
                yVals, yValsBad, axis=1
            )

            if np.all(np.logical_or(np.isinf(yValsMean), np.isnan(yValsMean))):
                break

            if showXValsSEM:
                if show_peaks_only:
                    xValsAllPrepA = np.array(
                        [v[np.newaxis, :] for v in res["peakPerfXValAll"]]
                    )
                    xValsAllPrep = np.array(res["peakPerfXValAll"][li][np.newaxis, :])
                else:
                    xValsAllPrepA = np.array(res["xValsAllPreMean"])
                    xValsAllPrep = np.array(res["xValsAllPreMean"][li])
                if removeBadsFromAll:
                    xValsAllPrepA = xValsAllPrepA[..., ~isBad]
                    xValsAllPrep = xValsAllPrep[:, ~isBad]
            else:
                xValsAllPrepA = np.array(xValsAll)
                xValsAllPrep = np.array(xValsAll[li])
            if xTakeLog2:
                xValsAllPrepA = np.log2(xValsAllPrepA)
                xValsAllPrep = np.log2(xValsAllPrep)
            if xTakeLog10:
                xValsAllPrepA = np.log10(xValsAllPrepA)
                xValsAllPrep = np.log10(xValsAllPrep)
            xValsThis = xValsAllPrep
            if perfVals2 is not None:
                xValsThis = yValsAll2[li % len(yValsAll2)]
                if yTakeLog2:
                    xValsThis = np.log2(xValsThis)
                if yTakeLog10:
                    xValsThis = np.log10(xValsThis)
                xValsThis = np.ones((yVals.shape[0], 1)) * xValsThis
                xValsThisExt = xValsThis
            elif showXValsSEM:
                xValsThisExt = xValsThis
            else:
                xValsThisExt = xValsThis[:, np.newaxis] @ np.ones((1, yVals.shape[1]))
            if showXValsSEM:
                xValsBad = np.logical_or(np.isnan(xValsThis), np.isinf(xValsThis))
                if np.any(xValsBad != yValsBad):
                    xValsBad = np.logical_or(xValsBad, yValsBad)
                    logger.info(
                        "Ignoring the {}/{} samples that are bad for either x or y".format(
                            np.sum(xValsBad), xValsBad.size
                        )
                    )
                    yValsMean, yValsMedian, yValsStd, yValsSEM = computeMaskedStats(
                        yVals, xValsBad, axis=1
                    )
                xValsMean, xValsMedian, xValsStd, xValsSEM = computeMaskedStats(
                    xValsThis, xValsBad, axis=1
                )
                plotXMean = xValsMean
            else:
                plotXMean = np.nanmean(xValsThis)

            finalXVals[li] = xValsThis

            col = colors[cInd % len(colors)]
            lStyle = linestyles[cInd % len(linestyles)]
            label = ""
            if showLetterGuides:
                letter = chr(ord("A") + cInd)
                label += "({}) ".format(letter)
            if len(legNames) > 0:
                label += legNames[int(np.min([li, len(legNames) - 1]))]
            else:
                label += yField
            if np.sum(yValsBad) > 0:
                label += f" [bad n={np.sum(yValsBad)}/{yValsBad.size}]"

            finalLabels[li] = label
            source_data.append({"x_data": xValsThis, "y_data": yVals, "label": label})
            lineMarker = lineMarkers[li % len(lineMarkers)]
            if showBox:
                x0, bar_dx = getBarPos(xValsThis, groupRelWith=0.5, li=li)
                h = ax.boxplot(
                    yVals.T,
                    positions=x0,
                    widths=0.8 * bar_dx,
                    patch_artist=True,
                    manage_ticks=False,
                )
                for bi in range(len(h["boxes"])):
                    h["boxes"][bi].set_color(col)
                    h["boxes"][bi].set_facecolor(col)
                    h["boxes"][bi].set_alpha(0.3)
                    h["medians"][bi].set_color((0, 0, 0))
                for bi in range(len(h["caps"])):
                    h["caps"][bi].set_color(col)
                    h["whiskers"][bi].set_color(col)
                if showDots:
                    for bi in range(len(h["fliers"])):
                        h["fliers"][bi].remove()
                    jitter_vals = (
                        (np.random.rand(yVals.shape[0], yVals.shape[1]) - 0.5)
                        * bar_dx
                        * 0.8
                        * 0.8
                    )
                    for xi in range(len(xValsThis)):
                        ax.scatter(
                            x0[xi] + jitter_vals[xi, :],
                            yVals[xi, :],
                            s=0.2,
                            facecolor=(0, 0, 0),
                            alpha=0.5,
                        )
                if writeMedian:
                    for xi in range(len(xValsThis)):
                        ax.annotate(
                            "{:.2g}".format(np.median(yVals[xi, :])),
                            (x0[xi], np.median(yVals[xi, :])),
                            color=col,
                        )
            if np.unique(xValsThis).size > 1 and not showXValsSEM:
                if showNoMeanLines == True:
                    labelThis = label if not showMeanLines else None
                    lWidth = 1.5 if not showMeanLines else 1
                    alpha = 1 if not showMeanLines else 0.4
                    ax.plot(
                        xValsThisExt,
                        yVals,
                        label=labelThis,
                        color=col,
                        linestyle=lStyle,
                        linewidth=lWidth,
                        alpha=alpha,
                        marker=lineMarker,
                    )
                if len(textValsNoMean) > 0:
                    yValsText = np.take(
                        textValsNoMean, res["groupInds"][li], axis=res["groupAxis"][li]
                    )  # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].
                    nonXAxes = np.where(
                        np.logical_not(
                            np.isin(range(len(textValsNoMean.shape)), xValAxis)
                        )
                    )[
                        0
                    ]  # Mean over these axes will be included in SEM
                    yValsText = np.transpose(
                        yValsText, tuple(np.concatenate((xValAxis, nonXAxes)))
                    )  # Bring the dimension that is not going to be averaged to the front
                    yValsText = np.reshape(
                        yValsText,
                        (yValsText.shape[0], np.prod(yValsText.shape[1:])),
                        order="F",
                    )
                    for yi in range(yVals.shape[1]):
                        for xi in range(len(xValsThis)):
                            if (
                                type(yValsText[xi, yi]) is str
                                and len(yValsText[xi, yi]) > 0
                            ):
                                ax.annotate(
                                    yValsText[xi, yi],
                                    (xValsThis[xi], yVals[xi, yi]),
                                    color=col,
                                )
                if showMeanLines:
                    if showSEM:
                        ax.fill_between(
                            xValsThis,
                            yValsMean - yValsSEM,
                            yValsMean + yValsSEM,
                            facecolor=col,
                            alpha=0.3,
                        )
                    ax.plot(
                        xValsThis,
                        yValsMean,
                        label=label,
                        color=col,
                        linewidth=1.5,
                        linestyle=lStyle,
                        marker=lineMarker,
                    )
                # source_data.append({
                #     'x_data': xValsThis,
                #     'y_data': yVals,
                #     'label': label
                # })
            elif not np.isnan(np.unique(xValsThis)[0]) and (
                not np.ma.isMA(plotXMean) or np.any(~plotXMean.mask)
            ):
                perfRangeRel = (np.max(yValsMean) - np.min(yValsMean)) / np.min(
                    yValsMean
                )
                if perfRangeRel > 1e-3 and "Time" not in yField:
                    logger.warning(
                        "WARNING: expected {} to have the same performance for the fixed nx of {}, but {} varied by {:.3g}%!".format(
                            label, np.mean(xValsThis), yField, perfRangeRel * 100
                        )
                    )
                if showNoMeanLines == True:
                    ax.scatter(
                        np.mean(xValsThis) * np.ones(yVals.shape[1]),
                        np.mean(yVals, axis=0),
                        c=col,
                        facecolor=col,
                        alpha=0.3,
                        marker=lineMarker,
                    )
                if showSEM:
                    if not showXValsSEM:
                        ax.axhspan(
                            np.mean(yValsMean) - np.mean(yValsSEM),
                            np.mean(yValsMean) + np.mean(yValsSEM),
                            alpha=0.25,
                            color=col,
                            linewidth=0,
                        )
                        xerr = None
                    else:
                        xerr = np.mean(xValsSEM)
                    ax.errorbar(
                        plotXMean,
                        np.mean(yValsMean),
                        yerr=np.mean(yValsSEM),
                        xerr=xerr,
                        color=col,
                        linestyle=lStyle,
                        alpha=0.8,
                        capsize=2,
                    )
                lineMarkerThis = lineMarker if lineMarkersBU is not None else "o"
                ax.scatter(
                    plotXMean,
                    np.mean(yValsMean),
                    s=20,
                    c=col,
                    facecolor=col,
                    label=label,
                    marker=lineMarkerThis,
                )
                if showDots and xValsThis.size == yVals.size:
                    ax.scatter(
                        xValsThis.flatten(),
                        yVals.flatten(),
                        s=0.2,
                        facecolor=col,
                        alpha=0.5,
                        marker=".",
                    )
                if showLetterGuides:
                    hvalignment = [
                        ("right", "bottom"),
                        ("right", "top"),
                        ("left", "bottom"),
                        ("left", "top"),
                    ]
                    ax.text(
                        plotXMean,
                        np.mean(yValsMean),
                        s=letter,
                        color=col,
                        horizontalalignment=hvalignment[li % len(hvalignment)][0],
                        verticalalignment=hvalignment[li % len(hvalignment)][1],
                        fontsize="xx-small",
                    )
                # source_data.append({
                #     'x_data': xValsThis,
                #     'y_data': yValsMean,
                #     'label': label
                # })
            if (
                showNoMeanLines == True
                and noMeanLabels is not None
                and len(noMeanLabels) == yVals.shape[1]
            ):
                for i in range(yVals.shape[1]):
                    ax.annotate(
                        noMeanLabels[i], (xValsThis[-1], yVals[-1, i]), color=col
                    )
            for si, scatterInd in enumerate(scatterInds):
                scatterLabel = None
                if len(scatterLabels) > 0 and li == 0 and yFieldInd == 0:
                    scatterLabel = scatterLabels[si % len(scatterLabels)]
                style = {
                    "markersize": None,
                    "marker": "o",
                    "color": col,
                    "linestyle": "-",
                }
                if isinstance(scatterStyles, (list, tuple, np.ndarray)):
                    if (
                        type(scatterStyles[si % len(scatterStyles)]) is not dict
                        and len(scatterStyles[si % len(scatterStyles)]) == 1
                    ):
                        style.update(scatterStyles[si % len(scatterStyles)][0])
                    elif type(scatterStyles[si % len(scatterStyles)]) is dict:
                        style.update(scatterStyles[si % len(scatterStyles)])
                elif type(scatterStyles) is dict:
                    style.update(scatterStyles)
                ax.plot(
                    xValsThis[scatterInd],
                    yValsMean[scatterInd],
                    label=scatterLabel,
                    markersize=style["markersize"],
                    lw=0,
                    c=style["color"],
                    fillstyle="none",
                    marker=style["marker"],
                )
            if write_peaks or write_peak_perfs:
                if np.unique(yValsMean).size > 1:
                    isOk = isAlmostAsGoodAsTheBest(
                        yValsMean[:, np.newaxis],
                        yValsSEM[:, np.newaxis],
                        yField,
                        criteria=peak_det_criteria,
                        peak_det_sem_multiplier=peak_det_sem_multiplier,
                        ratio=peak_det_ratio,
                    )
                    firstOkInd = np.where(isOk)[0][0]
                else:
                    firstOkInd = 0
                xy = (xValsThis[firstOkInd], yValsMean[firstOkInd])
                txt = ""
                if write_peaks:
                    txt += "{:.3g} ".format(xValsThis[firstOkInd])
                if write_peak_perfs:
                    txt += "{:.2g}".format(xy[1])
                ax.scatter(xy[0], xy[1], s=20, c=col, marker="x", linewidth=1.5)
                ax.annotate(
                    txt,
                    xy,
                    color=col,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

        if showStatTest and np.all([x is not None for x in finalXVals]):
            for li in range(1, len(yValsAll)):
                if statsVsFirst is True:
                    ind1, ind2 = 0, li
                else:
                    ind1, ind2 = li - 1, li
                x0, bar_dx0 = getBarPos(xValsAllPrepA[ind1], groupRelWith=0.5, li=ind1)
                x1, bar_dx1 = getBarPos(xValsAllPrepA[ind2], groupRelWith=0.5, li=ind2)
                for xi in range(len(xValsAllPrepA[ind2])):
                    try:
                        w, p = wilcoxon(
                            yValsAll[ind2][xi, :],
                            yValsAll[ind1][xi, :],
                            alternative="two-sided",
                        )
                    except Exception as e:
                        logger.info("Wilcoxon error: {}".format(e))
                        p = np.nan
                    cAxis = ax.axis()
                    y0 = cAxis[2] + (0.9 + 0.1 * ind2 / len(yValsAll)) * np.diff(
                        cAxis[2:]
                    )
                    xM = np.mean([x0[xi], x1[xi]])
                    ax.plot((x0[xi], x1[xi]), (y0, y0), c=(0, 0, 0), linewidth=0.5)
                    if p < 0.0005:
                        ax.annotate("{:.3g}\n***".format(p), (xM, y0), color=(0, 0, 0))
                    elif p < 0.005:
                        ax.annotate("{:.3g}\n**".format(p), (xM, y0), color=(0, 0, 0))
                    elif p < 0.05:
                        ax.annotate("{:.3g}\n*".format(p), (xM, y0), color=(0, 0, 0))
                    else:
                        ax.annotate("{:.3g}\nns".format(p), (xM, y0), color=(0, 0, 0))

        # if showStatTestLinePlot:
        #     addPairwiseColoredStatsLines(yValsAll, finalXVals, colors, linestyles, lineMarkers, ax=ax2)

        if show_peaks_only and highlight_fronter:
            # Find performance frontiers
            perfValsX = [x.flatten() for x in finalXVals if x is not None]
            if len(perfValsX) == 0:
                continue
            perfValsY = [y.flatten() for y in yValsAll]
            pValsX, pValNsX = computePairwiseStatsTests(perfValsX)
            pValsY, pValNsY = computePairwiseStatsTests(perfValsY)
            perfVals = [perfValsX, perfValsY]
            pairwisePVals = [pValsX, pValsY]
            perfNames = [yField, xValField]
            onFrontier = findPerformanceFrontier(
                perfVals,
                pairwisePVals,
                perfNames,
                min_relative_diff=frontier_min_relative_diff,
                labels=finalLabels,
            )
            for li in range(len(perfValsX)):
                if onFrontier[li]:
                    fInd = np.sum(onFrontier[:li])
                    xValsMean, xValsMedian, xValsStd, xValsSEM = computeMaskedStats(
                        perfValsX[li], yValsBad=None, axis=0
                    )
                    yValsMean, yValsMedian, yValsStd, yValsSEM = computeMaskedStats(
                        perfValsY[li], yValsBad=None, axis=0
                    )
                    ellipse = Ellipse(
                        (xValsMean, yValsMean),
                        np.max((2 * xValsSEM, 0.1 * float(np.diff(ax.get_xlim()))))
                        * (1 + fInd / 10),
                        np.max((2 * yValsSEM, 0.1 * float(np.diff(ax.get_ylim()))))
                        * (1 + fInd / 10),
                        angle=0,
                        alpha=1,
                        color=colors[li],
                        fill=False,
                        facecolor=None,
                        edgecolor=colors[li],
                        linewidth=1,
                        linestyle=["-", "--", "-.", ":"][fInd % 4],
                    )
                    ax.add_artist(ellipse)

    if (
        (XLim is None or len(XLim) == 0)
        and XLimTight
        and np.all([fXVal is not None for fXVal in finalXVals])
    ):
        XLim = [np.nanmin(np.array(finalXVals)), np.nanmax(np.array(finalXVals))]
    if XLim is not None and len(XLim) > 0 and ~np.any(np.isnan(np.array(XLim))):
        ax.set_xlim(XLim)
        if showStatTestLinePlot:
            ax2.set_xlim(XLim)
    if YLim is not None and len(YLim) > 0 and ~np.any(np.isnan(np.array(YLim))):
        ax.set_ylim(YLim)
    elif minYLim is not None:
        curYLim = list(ax.get_ylim())
        if curYLim[0] > minYLim[0]:
            curYLim[0] = minYLim[0]
        if curYLim[1] < minYLim[1]:
            curYLim[1] = minYLim[1]
        ax.set_ylim(curYLim)
    if plot45DegLine:
        xylims = np.concatenate(
            (np.atleast_2d(ax.get_xlim()), np.atleast_2d(ax.get_ylim())), axis=0
        )
        v = np.array([np.min(xylims[:, 0]), np.max(xylims[:, 1])])
        ax.plot(v, v, linestyle="-", color="#C0C0C0", alpha=0.5)
    if xLabel is None:
        if xValField is not None:
            xLabel = xValField
        else:
            xLabel = " ?"
        if xTakeLog10:
            xLabel = "Log10 {}".format(xLabel)
    ax.set_xlabel(xLabel)
    if yLabel is None:
        if len(yValFields) == 1:
            yLabel = yValFields[0]
        else:
            yLabel = " ? "
        if yTakeLog10:
            yLabel = "Log10 {}".format(yLabel)
    ax.set_ylabel(yLabel)
    if x_scale is not None:
        ax.set_xscale(x_scale, base=x_scale_base)
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if y_scale is not None:
        ax.set_yscale(y_scale, base=y_scale_base)
        ax.yaxis.set_major_formatter(ScalarFormatter())
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    ax.set_title("{}{}{}".format(titleHead, title, titleTail))
    font_size = (
        "xx-small"
        if len(legNames) > 0 and np.max([len(l) for l in legNames]) > 35
        else "x-small"
    )
    ax.legend(
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0,
        fontsize=font_size,
    )
    return_info = {
        "XLim": ax.get_xlim(),
        "YLim": ax.get_ylim(),
        "res": allRes,
        "res2": allRes2,
    }
    showOrSaveFig(fig, saveFile, saveExtensions)
    return return_info


def plotTimeSeriesPrediction(
    y_true,
    y_pred=None,
    t=None,
    t_pred=None,
    missing_marker=None,
    y_pred_is_list=False,
    shadeClass=None,
    showLegend=True,
    epochs=None,
    events=None,
    addNaNInTimeGaps=True,
    shortenTimeGaps=True,
    plotDims=[],
    dimNames=[],
    Ts=1,
    predPerfsToAdd=None,
    minYLim=None,
    YLim=[],
    XLim=[],
    YLimRelToTrueRange=None,
    xLabel=None,
    yLabel=None,
    colors=None,
    trueColor=None,
    predColor=None,
    predColor2=None,
    lineStyles=None,
    trueLegStrs=None,
    predLegStrs=None,
    titleHead="",
    title="",
    titleTail="",
    fig=None,
    figsize=(14, 8),
    skip_existing=False,
    saveFile=None,
    saveExtensions=None,
    return_fig=False,
):
    """Plots true and predicted values for a time series

    Args:
        y_true (_type_): _description_
        y_pred (_type_, optional): _description_. Defaults to None.
        t (_type_, optional): _description_. Defaults to None.
        t_pred (_type_, optional): _description_. Defaults to None.
        missing_marker (_type_, optional): _description_. Defaults to None.
        shadeClass (_type_, optional): _description_. Defaults to None.
        showLegend (bool, optional): _description_. Defaults to True.
        events (dict, optional): _description_. Defaults to {}.
        addNaNInTimeGaps (bool, optional): _description_. Defaults to True.
        shortenTimeGaps (bool, optional): _description_. Defaults to True.
        plotDims (list, optional): _description_. Defaults to [].
        dimNames (list, optional): _description_. Defaults to [].
        Ts (int, optional): _description_. Defaults to 1.
        predPerfsToAdd (_type_, optional): _description_. Defaults to None.
        minYLim (_type_, optional): _description_. Defaults to None.
        YLim (list, optional): _description_. Defaults to [].
        XLim (list, optional): _description_. Defaults to [].
        xLabel (_type_, optional): _description_. Defaults to None.
        yLabel (_type_, optional): _description_. Defaults to None.
        colors (_type_, optional): _description_. Defaults to None.
        trueColor (_type_, optional): _description_. Defaults to None.
        predColor (_type_, optional): _description_. Defaults to None.
        predColor2 (_type_, optional): _description_. Defaults to None.
        titleHead (str, optional): _description_. Defaults to ''.
        title (str, optional): _description_. Defaults to ''.
        titleTail (str, optional): _description_. Defaults to ''.
        fig (_type_, optional): _description_. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (14,8).
        skip_existing (bool, optional): _description_. Defaults to False.
        saveFile (_type_, optional): _description_. Defaults to None.
        saveExtensions (_type_, optional): _description_. Defaults to None.
        return_fig (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if skip_existing and checkIfAllExtsAlreadyExist(saveFile, saveExtensions):
        logger.info("Skipping... figure already exists: " + saveFile)
        return
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    if trueColor is None:
        trueColor = colors[0]
    if predColor is None:
        predColor = colors[1:]
    if predColor2 is None:
        predColor2 = colors[2:]
    if not isinstance(predColor, (list)):
        predColor = [predColor]
    if lineStyles is None:
        lineStyles = ["-"]
    if trueLegStrs is None:
        trueLegStrs = ["True"]
    if predLegStrs is None:
        predLegStrs = ["Pred"]
    if predPerfsToAdd is None:
        predPerfsToAdd = []
    if not isinstance(y_true, (list, tuple)):
        if len(y_true.shape) == 1:
            y_true = np.array(y_true[:, np.newaxis])
        elif len(y_true.shape) > 2:
            y_true = np.reshape(y_true, (y_true.shape[0], -1), "F")
        y_true = copy.copy(np.split(y_true, y_true.shape[1], axis=1))
    y_preds = [copy.deepcopy(y_pred)] if not y_pred_is_list else copy.deepcopy(y_pred)
    for i, y_pred in enumerate(y_preds):
        if y_pred is not None and not isinstance(y_pred, (list, tuple)):
            if len(y_pred.shape) == 1:
                y_pred = np.array(y_pred[:, np.newaxis])
            elif len(y_pred.shape) > 2:
                y_pred = np.reshape(y_pred, (y_pred.shape[0], -1), "F")
            y_pred = copy.copy(np.split(y_pred, y_pred.shape[1], axis=1))
        y_preds[i] = y_pred
    if t is not None and not isinstance(t, (list, tuple)):
        if len(t.shape) == 1:
            t = np.array(t[:, np.newaxis])
        t = copy.copy(np.split(t, t.shape[1], axis=1))
    T = copy.copy(t)

    if t_pred is not None and not isinstance(t_pred, (list, tuple)):
        if len(t_pred.shape) == 1:
            t_pred = np.array(t_pred[:, np.newaxis])
        t_pred = copy.copy(np.split(t_pred, t_pred.shape[1], axis=1))
    if t_pred is None:
        t_pred = T

    if (
        events is None
        and epochs is not None
        and len(epochs) > 0
        and "events" in epochs[0]
    ):
        events = {}
        for e in epochs:
            for k, v in e["events"].items():
                if k not in events:
                    events[k] = []
                events[k].append(v)

    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        plt.figure(fig)
        figsize = fig.get_size_inches()
    fig.subplots_adjust(
        hspace=0.1, wspace=0, top=0.89, bottom=0.05, left=0.05, right=0.8
    )
    if plotDims is None or len(plotDims) == 0:
        plotDims = range(len(y_true))
    if dimNames is None or len(dimNames) == 0:
        dimNames = ["Dim{}".format(yi) for yi in range(len(y_true))]
    for axInd, yi in enumerate(plotDims):
        ax = fig.add_subplot(len(plotDims), 1, 1 + axInd)
        prepAxesStyle(ax)
        if T is None:
            t = np.expand_dims(np.arange(y_true[yi].shape[0]) * Ts, 1)
        else:
            t = np.array(T[yi % len(T)])
        tCopy = np.array(t)
        yVal = np.array(y_true[yi], dtype=float)
        if len(yVal.shape) > 2:
            yVal = np.reshape(yVal, (yVal.shape[0], -1), "F")
        if shortenTimeGaps:
            dt = np.median(np.diff(t, axis=0))
            t, timeRemapper = shortenGaps(t, max_gap=10 * dt)
        else:
            timeRemapper = None
        if addNaNInTimeGaps:
            preGapInds = getGapInds(t)
            if len(preGapInds):
                t = np.insert(t, 1 + preGapInds, np.nan, axis=0)
                yVal = np.insert(yVal, 1 + preGapInds, np.nan, axis=0)
        if missing_marker is not None:
            yVal[yVal == missing_marker] = np.nan
            # add a point after lonely points to make sure they are plotted
            lonely_ind = (
                1
                + np.where(
                    np.logical_and(
                        ~np.isnan(yVal[1:-1]),
                        np.logical_and(np.isnan(yVal[:-2]), np.isnan(yVal[2:])),
                    )
                )[0]
            )
            yValNew = np.insert(yVal, 1 + lonely_ind, yVal[lonely_ind, 0])[
                :, np.newaxis
            ]
            tDiff = np.diff(t[:, 0], axis=0)
            tNew = np.insert(
                np.array(t[:, 0], np.floating),
                1 + lonely_ind,
                t[lonely_ind, 0] + 0.1 * tDiff[lonely_ind],
            )
            # np.concatenate((tNew[:, np.newaxis],yValNew[:, np.newaxis]), axis=1) # You can check this as a test
            t = tNew[:, np.newaxis]
            yVal = yValNew
        h1 = ax.plot(
            t, yVal, color=trueColor, linestyle=lineStyles[0 % len(lineStyles)]
        )
        thisTitle = "{}".format(dimNames[yi])
        legStrs = [trueLegStrs[0]]
        for listInd, y_pred in enumerate(y_preds):
            if y_pred is not None:
                yVal_pred = np.array(y_pred[yi], dtype="float")
                if len(yVal_pred.shape) > 2:
                    yVal_pred = np.reshape(yVal_pred, (yVal_pred.shape[0], -1), "F")
                if t_pred is None:
                    t_pred_this = tCopy
                else:
                    t_pred_this = t_pred[yi % len(t_pred)]
                if timeRemapper is not None:
                    t_pred_this = timeRemapper.apply(t_pred_this)
                if addNaNInTimeGaps and len(preGapInds):
                    t_pred_this = np.insert(t_pred_this, 1 + preGapInds, np.nan, axis=0)
                    yVal_pred = np.insert(yVal_pred, 1 + preGapInds, np.nan, axis=0)
                h2 = ax.plot(
                    t_pred_this,
                    yVal_pred,
                    color=predColor[listInd % len(predColor)],
                    linestyle=lineStyles[(1 + listInd) % len(lineStyles)],
                )
                thisLeg = predLegStrs[listInd % len(predLegStrs)]
                if (
                    len(predPerfsToAdd) > 0
                    and y_pred[yi] is not None
                    and yVal.shape[0] == yVal_pred.shape[0]
                ):
                    perfs = evaluateDecoding(yVal, yVal_pred, measures=predPerfsToAdd)
                    perfsStr = ", ".join(
                        [
                            "\n{}={:.3g}".format(pm, perfs["mean" + pm])
                            for pm in predPerfsToAdd
                        ]
                    )
                    if len(y_pred) > 1:
                        thisLeg += " " + perfsStr
                    else:
                        thisTitle += f" [{listInd}]" if listInd > 0 else " " + perfsStr
                legStrs.append(thisLeg)
        if axInd == 0:
            ax.set_title("{}{}{}\n{}".format(titleHead, title, titleTail, thisTitle))
        else:
            ax.set_title("{}".format(thisTitle))
        if axInd < len(plotDims) - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", width=0)
            ax.spines["bottom"].set_visible(False)
        if showLegend and legStrs is not None:
            ax.legend(
                legStrs,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0,
                fontsize="x-small",
            )
        if XLim is not None and len(XLim) > 0:
            if timeRemapper is not None:
                ax.set_xlim(timeRemapper.apply(np.array(XLim)))
            else:
                ax.set_xlim(XLim)
        if YLim is not None and len(YLim) > 0:
            ax.set_ylim(YLim)
        elif minYLim is not None:
            curYLim = list(ax.get_ylim())
            if curYLim[0] > minYLim[0]:
                curYLim[0] = minYLim[0]
            if curYLim[1] < minYLim[1]:
                curYLim[1] = minYLim[1]
            ax.set_ylim(curYLim)
        elif YLimRelToTrueRange is not None:
            trueRange = [np.nanmin(yVal), np.nanmax(yVal)]
            thisYLim = [
                trueRange[0] + np.diff(trueRange) * YLimRelToTrueRange[0],
                trueRange[1] + np.diff(trueRange) * YLimRelToTrueRange[1],
            ]
            ax.set_ylim(thisYLim)
        # if shadeClass is not None:
        #     addShadedAreasToXAxis(shadeClass, t)
        if events is not None:
            curYLim = list(ax.get_ylim())
            try:
                for ki, key in enumerate(events.keys()):
                    vals = np.array(events[key])
                    if not isinstance(events[key][0], (float, int)):
                        continue
                    vals = vals[~np.isnan(vals)]
                    if timeRemapper is not None:
                        vals = timeRemapper.apply(vals)
                    inRange = np.nonzero(np.logical_and(vals >= t[0], vals <= t[-1]))[0]
                    if inRange.size > 0:
                        col = colors[ki % len(colors)]
                        ax.vlines(
                            vals[inRange], curYLim[0], curYLim[1], alpha=0.5, colors=col
                        )
                        if axInd == 0:
                            ax.text(
                                x=vals[inRange[ki % inRange.size]],
                                y=curYLim[-1],
                                s=key,
                                color=col,
                                rotation="vertical",
                                horizontalalignment="right",
                                verticalalignment="top",
                                fontweight="bold",
                            )
            except Exception as e:
                logger.info(e)

    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
    if return_fig:
        return fig
    showOrSaveFig(fig, saveFile, saveExtensions)


def checkIfAllExtsAlreadyExist(saveFile, saveExtensions):
    """Checks if all expected extensions of a save file exist

    Args:
        saveFile (_type_): _description_
        saveExtensions (_type_): _description_

    Returns:
        _type_: _description_
    """
    if saveFile is not None:
        if saveExtensions is None:
            saveExtensions = [""]
        allExist = True
        for ext in saveExtensions:
            if len(ext) > 0 and ext[0] != ".":
                ext = "." + ext
            saveFileThis = "{}{}".format(saveFile, ext)
            allExist = allExist and os.path.exists(saveFileThis)
    else:
        allExist = False
    return allExist


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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="in", length=2)


def prepAxisForSaving():
    """Prepares some matplotlib axis format settings to be used for all saved files"""
    plt.rcParams["svg.fonttype"] = (
        "none"  # Assume fonts are installed on the machine where the SVG will be viewed.
    )
    plt.rcParams["pdf.use14corefonts"] = True  # Embeds texts as text, rather than paths
    # try:
    #     plt.rcParams['font.family'] = 'Times New Roman Reg'
    # except Exception as e:
    #     plt.rcParams['font.family'] = 'Times New Roman'  # Loading is buggy for the default Times font and always loads bold
    #     print(e)
    fsize = 10
    plt.rc("font", size=fsize)
    plt.rc("axes", titlesize=fsize)  # fontsize of the axes title
    plt.rc("axes", labelsize=fsize)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=fsize)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=fsize)  # fontsize of the tick labels
    plt.rc("legend", fontsize=fsize)  # legend fontsize
    plt.rc("figure", titlesize=fsize)  # fontsize of the figure title


def showOrSaveFig(
    fig, saveFile=None, saveExtensions=None, rotate_axes=True, rotation_specs=None
):
    """Shows or saves a generated matplotlib figure

    Args:
        fig (_type_): _description_
        saveFile (_type_, optional): _description_. Defaults to None.
        saveExtensions (_type_, optional): _description_. Defaults to None.
        rotation_specs (_type_, optional): _description_. Defaults to None.
    """
    prepAxisForSaving()
    if saveFile is None:
        plt.show()
    else:
        Path(saveFile).parent.mkdir(parents=True, exist_ok=True)
        if saveExtensions is None:
            saveExtensions = [""]
        for ext in saveExtensions:
            if len(ext) > 0 and ext[0] != ".":
                ext = "." + ext
            saveFileThis = "{}{}".format(saveFile, ext)
            try:
                plt.savefig(saveFileThis)
                logger.info("Figure saved as {}".format(saveFileThis))
            except Exception as e:
                logger.warning(
                    "Saving as {} failed \nError: {}".format(saveFileThis, e)
                )
        plt.close(fig)
        del fig
