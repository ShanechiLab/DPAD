import numpy as np
from PSID import LSSM
from PSID.PSID import projOrth

from .. import DPADModel
from .plot import checkIfAllExtsAlreadyExist, plotPredictionScatter
from .SSM import SSM
from .tools import pickColumnOp


def plot_model_params(
    sId,
    Z,
    Y,
    X,
    title,
    savePath=None,
    saveExtensions=["png", "svg"],
    skip_existing=False,
    trueModel=None,
    ZTrue=None,
    YTrue=None,
    XTrue=None,
    params_to_plot=None,
    plot_orig=None,
    figsize=None,
    show_hist_x=False,
    show_hist_y=False,
    XLimByQuantile=None,
    YLimByQuantile=None,
    x_keep_quantiles=None,
):
    """Plots parameters of the learned model

    Args:
        sId (object): learned model
        Z (np.array): signal 2 (e.g. behavior)
        Y (np.array): signal 1 (e.g. neural activity)
        X (np.array): latent state
        title (string): title of figure
        savePath (string, optional): path to save figure files. Defaults to None.
        saveExtensions (list, optional): list of figure file extensions to generate. Defaults to ['png', 'svg'].
        skip_existing (bool, optional): if True will skip generating plots if the figure file exists. Defaults to False.
        trueModel (object, optional): true model if known in simulations. Defaults to None.
        ZTrue (np.array, optional): true values of signal 2 (e.g. behavior) if known in simulations. Defaults to None.
        YTrue (np.array, optional): true values of signal 1 (e.g. neural activity) if known in simulations. Defaults to None.
        XTrue (np.array, optional): true values of latent states if known in simulations. Defaults to None.
        params_to_plot (list of string, optional): list of parameters to plot. Defaults to None.
        plot_orig (bool, optional): if True, will try to plot true parameter values in simulations when possible. Defaults to None.
        # The following are arguments to pass to plotPredictionScatter
        figsize (_type_, optional): _description_. Defaults to None.
        show_hist_x (bool, optional): _description_. Defaults to False.
        show_hist_y (bool, optional): _description_. Defaults to False.
        XLimByQuantile (_type_, optional): _description_. Defaults to None.
        YLimByQuantile (_type_, optional): _description_. Defaults to None.
        x_keep_quantiles (_type_, optional): _description_. Defaults to None.
    """
    if plot_orig is None:
        plot_orig = XTrue is None
    nx = X.shape[-1]
    ny = Y.shape[-1]
    if isinstance(sId, LSSM) and not isinstance(sId, SSM):
        sId_DPAD = DPADModel()
        sId_DPAD.setToLSSM(sId, model1_Cy_Full=False, model2_Cz_Full=False)
        sId = sId_DPAD
    if isinstance(sId, DPADModel):
        if sId.model1 is not None:
            X1 = X[:, : sId.n1]
            X2 = X[:, sId.n1 :]
            XRng = range(X.shape[1])
            X1Rng = range(X1.shape[1])
            YRng = range(Y.shape[1])
            X1Cols = [(X1 @ pickColumnOp(X1.shape[1], [xi])).T for xi in X1Rng]
            YCols = [(Y @ pickColumnOp(Y.shape[1], [yi])).T for yi in YRng]

            if XTrue is not None:
                zDims = trueModel.zDims
                n1True = zDims.size
                X1True = XTrue[:, :n1True]
                X2True = XTrue[:, n1True:]
                X1TrueCols = [
                    (X1True @ pickColumnOp(X1True.shape[1], [xi])).T for xi in X1Rng
                ]

                X1TrueHat, WToX1True = projOrth(X1True.T, X1.T)
                X1TrueHat = X1TrueHat.T
                # X1TrueHat is the estimated equivalent X1True for every identified X1.
                # We expect true_param(X1TrueHat) to be the same as learned_param(X1)

                X1Hat, WToX1 = projOrth(X1.T, X1True.T)
                X1Hat = X1Hat.T
                # X1Hat is the estimated equivalent identified X1 for every X1True.
                # We expect learned_param(X1Hat) to be the same as true_param(X1True)

                X1TrueCols = [
                    (X1True @ pickColumnOp(X1True.shape[1], [xi])).T for xi in X1Rng
                ]
                XTrueCols = [
                    (XTrue @ pickColumnOp(XTrue.shape[1], [xi])).T for xi in XRng
                ]
            else:
                X1Hat = X1
            X1HatCols = [(X1Hat @ pickColumnOp(X1Hat.shape[1], [xi])).T for xi in X1Rng]
            if (
                (params_to_plot is None or "A" in params_to_plot)
                and hasattr(sId.model1.rnn.cell, "A")
                and (
                    "unifiedAK" not in sId.model1.cell_args
                    or not sId.model1.cell_args["unifiedAK"]
                )
            ):
                if plot_orig and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramA_KCy", saveExtensions
                    )
                    or not skip_existing
                ):
                    y_out_A_list = [
                        sId.model1.rnn.cell.A.predict(X1Cols[xi]).T for xi in X1Rng
                    ]
                    plotPredictionScatter(
                        [
                            np.tile(X1[:, xi].T, (y_out_A_list[0].shape[1], 1))
                            for xi in X1Rng
                        ],
                        [y_out_A_list[xi].T for xi in X1Rng],
                        connect_sorted=True,
                        square=False,
                        styles={"size": 1},
                        title=f"{title}A" "=(A-KCy) stage 1",
                        xLabel=[f"x{xi+1}" for xi in X1Rng],
                        yLabel=[f"A'(x{xi+1})" for xi in X1Rng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=None if savePath is None else savePath + "_paramA_KCy",
                        saveExtensions=saveExtensions,
                    )
                if XTrue is not None and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramA_KCy_sim", saveExtensions
                    )
                    or not skip_existing
                ):
                    y_out_A_list = [
                        sId.model1.rnn.cell.A.predict(X1HatCols[xi]).T for xi in X1Rng
                    ]
                    if trueModel is not None:
                        if isinstance(trueModel, SSM):
                            # y_out_A_true = trueModel.apply_param('A', XTrue.T).T \
                            #             - trueModel.apply_param('K', trueModel.apply_param('C', XTrue.T)).T
                            y_out_A_true = [
                                trueModel.apply_param("A_KC", XTrueCols[xi]).T
                                for xi in X1Rng
                            ]
                        else:
                            y_out_A_true = [
                                (trueModel.A_KC @ XTrueCols[xi]).T for xi in X1Rng
                            ]
                        # Convert back to the basis of the learned model
                        y_out_A_true_sim = [
                            (WToX1 @ y_out_A_true[xi][:, :n1True].T).T for xi in X1Rng
                        ]
                        # y_out_A_true_sim, W2 = projOrth(y_out_A.T, y_out_A_true.T) # TEMP
                        # y_out_A_true_sim = y_out_A_true_sim.T
                        y_out_A_list = [
                            np.concatenate(
                                (y_out_A_list[xi], y_out_A_true_sim[xi]), axis=1
                            )
                            for xi in X1Rng
                        ]
                    plotPredictionScatter(
                        [
                            np.tile(X1True[:, xi].T, (y_out_A_list[0].shape[1], 1))
                            for xi in X1Rng
                        ],
                        [y_out_A_list[xi].T for xi in X1Rng],
                        connect_sorted=True,
                        legNames=["Learned", "True (sim)"],
                        square=False,
                        styles=[{"linestyle": "-"}, {"linestyle": "--"}],
                        title=f"{title}A" "=(A-KCy) stage 1",
                        xLabel=[f"x{xi+1}" for xi in X1Rng],
                        yLabel=[f"A" "(x{xi+1})" for xi in X1Rng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=(
                            None if savePath is None else savePath + "_paramA_KCy_sim"
                        ),
                        saveExtensions=saveExtensions,
                    )
            if (params_to_plot is None or "Cz" in params_to_plot) and hasattr(
                sId.model1.rnn.cell, "C"
            ):
                if plot_orig and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramCz", saveExtensions
                    )
                    or not skip_existing
                ):
                    z_out_Cz_list = [
                        sId.model1.rnn.cell.C.predict(X1Cols[xi]).T for xi in X1Rng
                    ]
                    plotPredictionScatter(
                        [
                            np.tile(X1[:, xi].T, (z_out_Cz_list[0].shape[1], 1))
                            for xi in X1Rng
                        ],
                        [z_out_Cz_list[xi].T for xi in X1Rng],
                        connect_sorted=True,
                        square=False,
                        styles={"size": 1},
                        title=[f"{title}Cz1{xi+1}" for xi in X1Rng],
                        xLabel=[f"x{xi+1}" for xi in X1Rng],
                        yLabel=[f"Cz(x{xi+1})" for xi in X1Rng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=None if savePath is None else savePath + "_paramCz",
                        saveExtensions=saveExtensions,
                    )
                if XTrue is not None and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramCz_sim", saveExtensions
                    )
                    or not skip_existing
                ):
                    z_out_Cz_list = [
                        sId.model1.rnn.cell.C.predict(X1HatCols[xi]).T for xi in X1Rng
                    ]
                    if trueModel is not None:  # Add true model's param
                        if isinstance(trueModel, SSM):
                            z_out_Cz_true = [
                                trueModel.apply_param("Cz", XTrueCols[xi]).T
                                for xi in X1Rng
                            ]
                        else:
                            z_out_Cz_true = [
                                (trueModel.Cz @ XTrueCols[xi]).T for xi in X1Rng
                            ]
                        z_out_Cz_list = [
                            np.concatenate(
                                (z_out_Cz_list[xi], z_out_Cz_true[xi]), axis=1
                            )
                            for xi in X1Rng
                        ]
                    plotPredictionScatter(
                        [
                            np.tile(X1True[:, xi].T, (z_out_Cz_list[0].shape[1], 1))
                            for xi in X1Rng
                        ],
                        [z_out_Cz_list[xi].T for xi in X1Rng],
                        connect_sorted=True,
                        legNames=["Learned", "True (sim)"],
                        square=False,
                        styles=[{"linestyle": "-"}, {"linestyle": "--"}],
                        title=[f"{title}Cz1{xi+1}" for xi in X1Rng],
                        xLabel=[f"x{xi+1}" for xi in X1Rng],
                        yLabel=[f"Cz(x{xi+1})" for xi in X1Rng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=(
                            None if savePath is None else savePath + "_paramCz_sim"
                        ),
                        saveExtensions=saveExtensions,
                    )
            if (params_to_plot is None or "K" in params_to_plot) and hasattr(
                sId.model1.rnn.cell, "K"
            ):
                if plot_orig and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramK1", saveExtensions
                    )
                    or not skip_existing
                ):
                    y_out_K_list = [
                        sId.model1.rnn.cell.K.predict(YCols[yi]).T for yi in YRng
                    ]
                    plotPredictionScatter(
                        [
                            np.tile(Y[:, yi].T, (y_out_K_list[0].shape[1], 1))
                            for yi in YRng
                        ],
                        [y_out_K_list[yi].T for yi in YRng],
                        connect_sorted=True,
                        square=False,
                        styles={"size": 1},
                        title=[f"{title}K1{yi+1}" for yi in YRng],
                        xLabel=[f"y{yi+1}" for yi in YRng],
                        yLabel=[f"K1(y{yi+1})" for yi in YRng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=None if savePath is None else savePath + "_paramK1",
                        saveExtensions=saveExtensions,
                    )
                if XTrue is not None and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramK1_sim", saveExtensions
                    )
                    or not skip_existing
                ):
                    y_out_K_list = [
                        (WToX1True @ sId.model1.rnn.cell.K.predict(YCols[yi])).T
                        for yi in YRng
                    ]
                    if trueModel is not None:  # Add true model's param
                        if isinstance(trueModel, SSM):
                            y_out_K_true = [
                                trueModel.apply_param("K", YCols[yi]).T for yi in YRng
                            ]
                        else:
                            y_out_K_true = [(trueModel.K @ YCols[yi]).T for yi in YRng]
                        y_out_K_true_X1 = [y_out_K_true[yi][:, :n1True] for yi in YRng]
                        y_out_K_list = [
                            np.concatenate(
                                (y_out_K_list[yi], y_out_K_true_X1[yi]), axis=1
                            )
                            for yi in YRng
                        ]
                    plotPredictionScatter(
                        [
                            np.tile(Y[:, yi].T, (y_out_K_list[0].shape[1], 1))
                            for yi in YRng
                        ],
                        [y_out_K_list[yi].T for yi in YRng],
                        connect_sorted=True,
                        legNames=[
                            f"Learned (x{xi+1})" for xi in range(WToX1True.shape[0])
                        ]
                        + [f"True (x{xi+1}) (sim)" for xi in range(WToX1True.shape[0])],
                        square=False,
                        styles=[{"linestyle": "-"}] * WToX1True.shape[0]
                        + [{"linestyle": "--"}] * WToX1True.shape[0],
                        title=[f"{title}K1{yi+1}" for yi in YRng],
                        xLabel=[f"y{yi+1}" for yi in YRng],
                        yLabel=[f"K1(y{yi+1})" for yi in YRng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=(
                            None if savePath is None else savePath + "_paramK1_sim"
                        ),
                        saveExtensions=saveExtensions,
                    )
            if params_to_plot is None or "Cy" in params_to_plot:
                if plot_orig and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramCy", saveExtensions
                    )
                    or not skip_existing
                ):
                    y_out_Cy_list = [
                        sId.model1_Cy.predict(X1Cols[xi]).T for xi in X1Rng
                    ]
                    plotPredictionScatter(
                        [
                            np.tile(X1[:, xi].T, (y_out_Cy_list[0].shape[1], 1))
                            for xi in X1Rng
                        ],
                        [y_out_Cy_list[xi].T for xi in X1Rng],
                        connect_sorted=True,
                        square=False,
                        styles={"size": 1},
                        title=[f"{title}Cy1{xi+1}" for xi in X1Rng],
                        xLabel=[f"x{xi+1}" for xi in X1Rng],
                        yLabel=[f"Cy(x{xi+1})" for xi in X1Rng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=None if savePath is None else savePath + "_paramCy",
                        saveExtensions=saveExtensions,
                    )
                if XTrue is not None and (
                    savePath is None
                    or not checkIfAllExtsAlreadyExist(
                        savePath + "_paramCy_sim", saveExtensions
                    )
                    or not skip_existing
                ):
                    y_out_Cy_list = [
                        sId.model1_Cy.predict(X1HatCols[xi]).T for xi in X1Rng
                    ]
                    if trueModel is not None:  # Add true model's param
                        if isinstance(trueModel, SSM):
                            y_out_Cy_true = [
                                trueModel.apply_param("C", XTrueCols[xi]).T
                                for xi in X1Rng
                            ]
                        else:
                            y_out_Cy_true = [
                                (trueModel.C @ XTrueCols[xi]).T for xi in X1Rng
                            ]
                        y_out_Cy_list = [
                            np.concatenate(
                                (y_out_Cy_list[xi], y_out_Cy_true[xi]), axis=1
                            )
                            for xi in X1Rng
                        ]
                    plotPredictionScatter(
                        [
                            np.tile(X1True[:, xi].T, (y_out_Cy_list[0].shape[1], 1))
                            for xi in X1Rng
                        ],
                        [y_out_Cy_list[xi].T for xi in X1Rng],
                        connect_sorted=True,
                        legNames=["Learned", "True (sim)"],
                        square=False,
                        styles=[{"linestyle": "-"}, {"linestyle": "--"}],
                        title=[f"{title}Cy1{xi+1}" for xi in X1Rng],
                        xLabel=[f"x{xi+1}" for xi in X1Rng],
                        yLabel=[f"Cy(x{xi+1})" for xi in X1Rng],
                        figsize=figsize,
                        show_hist_x=show_hist_x,
                        show_hist_y=show_hist_y,
                        XLimByQuantile=XLimByQuantile,
                        YLimByQuantile=YLimByQuantile,
                        x_keep_quantiles=x_keep_quantiles,
                        skip_existing=skip_existing,
                        saveFile=(
                            None if savePath is None else savePath + "_paramCy_sim"
                        ),
                        saveExtensions=saveExtensions,
                    )
                    pass
