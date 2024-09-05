""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""An SSM object for keeping parameters, filtering, etc"""
import logging
import time

import numpy as np
from PSID.LSSM import LSSM, genRandomGaussianNoise
from sympy import Poly, symbols
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SSM(LSSM):
    def __init__(self, lssm=None, **kwargs):
        if lssm is not None:
            if "params" not in kwargs:
                kwargs["params"] = {}
            for p in lssm.getListOfParams():
                kwargs["params"][p] = getattr(lssm, p)
        super().__init__(**kwargs)

    def changeParamsIsolated(self, params={}):
        """
        Changes given parameters but DOES NOT update any other potentially dependent parameters!
        Use with care!
        """
        for k, v in params.items():
            setattr(self, k, v)

    def get_pram_sym(self, param_name):
        """
        Returns a symbolic var reprenting the operation of one parameter
        """
        p = getattr(self, param_name)

        n_out, n_in = p.shape
        in_sym_names = ",".join(["x{}".format(ii) for ii in range(n_in)])
        in_syms = symbols(in_sym_names)
        if not isinstance(in_syms, tuple):
            in_syms = (in_syms,)
        symP = []
        for oi in range(n_out):
            dimP = 0
            for ii in range(n_in):
                dimP += float(p[oi, ii]) * in_syms[ii]
            symP.append(dimP.as_poly())
        return symP

    def apply_param(self, param_name, input, **kwargs):
        p = getattr(self, param_name)
        if isinstance(p, np.ndarray) and not isinstance(p.flatten()[0], Poly):
            out = p @ input
        elif isinstance(np.array(p).flatten()[0], Poly):
            pFlat = np.array(p).flatten()
            out = np.empty((len(pFlat), input.shape[1]))
            for dimInd, dimP in enumerate(pFlat):
                dimPExpr = dimP.as_expr()
                var_syms = dimPExpr.free_symbols
                var_names = list(map(str, var_syms))
                var_syms_sorted = [v2 for v1, v2 in sorted(zip(var_names, var_syms))]
                if isinstance(dimP, Poly):
                    timeInds = range(input.shape[1])
                    if input.shape[1] > 100:
                        timeInds = tqdm(timeInds, f"applying {param_name}")
                    for ti in timeInds:
                        subVarVals = input[:, ti]
                        res = dimPExpr.subs(list(zip(var_syms_sorted, subVarVals)))
                        out[dimInd, ti] = float(res)
        else:
            raise (Exception("Not supported"))
        return out

    def get_param_io_count(self, param_name):
        p = getattr(self, param_name)
        if isinstance(p, np.ndarray) and not isinstance(p.flatten()[0], Poly):
            in_dim = p.shape[-1]
            out_dim = p.shape[0]
        elif isinstance(np.array(p).flatten()[0], Poly):
            pFlat = np.array(p).flatten()
            in_dim = len((pFlat[0].as_expr()).free_symbols)
            out_dim = len(pFlat)
        else:
            raise (Exception("Not supported"))
        return in_dim, out_dim

    def generateObservationFromStates(
        self, X, u=None, param_names=["C", "D"], prep_model_param="", mapping_param=""
    ):
        Y = None
        if hasattr(self, param_names[0]):
            C = getattr(self, param_names[0])
        else:
            C = None
        if len(param_names) > 1 and hasattr(self, param_names[1]):
            D = getattr(self, param_names[1])
        else:
            D = None

        if len(param_names) > 2 and hasattr(self, param_names[2]):
            errSys = getattr(self, param_names[2])
        else:
            errSys = None

        if C is not None or D is not None:
            ny = (
                self.get_param_io_count(param_names[0])[1]
                if C is not None
                else self.get_param_io_count(param_names[1])[1]
            )
            N = X.shape[0]
            Y = np.zeros((N, ny))
            if C is not None:
                Y += self.apply_param(param_names[0], X.T).T
            if D is not None and u is not None:
                if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                    u = self.UPrepModel.apply(
                        u, time_first=True
                    )  # Apply any mean removal/zscoring
                Y += self.apply_param(param_names[1], u.T).T

        if errSys is not None:
            err = errSys.generateRealization(N=N, return_z=True)[2]
            if err is not None:
                Y = Y + err if Y is not None else Y

        if prep_model_param is not None and hasattr(self, prep_model_param):
            prep_model_param_obj = getattr(self, prep_model_param)
            if prep_model_param_obj is not None:
                Y = prep_model_param_obj.apply_inverse(
                    Y
                )  # Apply inverse of any mean-removal/zscoring

        if mapping_param is not None and hasattr(self, mapping_param):
            mapping_param_obj = getattr(self, mapping_param)
            if mapping_param_obj is not None and hasattr(mapping_param_obj, "map"):
                Y = mapping_param_obj.map(Y)
        return Y

    def generateRealizationWithQRS(
        self,
        N,
        x0=None,
        w0=None,
        u0=None,
        u=None,
        wv=None,
        return_z=False,
        return_z_err=False,
        return_wv=False,
        blowup_threshold=None,
        reset_x_on_blowup=None,
        randomize_x_on_blowup=None,
    ):
        if blowup_threshold is None:
            if hasattr(self, "blowup_threshold"):
                blowup_threshold = self.blowup_threshold
            else:
                blowup_threshold = np.inf
        if reset_x_on_blowup is None:
            if hasattr(self, "reset_x_on_blowup"):
                reset_x_on_blowup = self.reset_x_on_blowup
            else:
                reset_x_on_blowup = False
        if randomize_x_on_blowup is None:
            if hasattr(self, "randomize_x_on_blowup"):
                randomize_x_on_blowup = self.randomize_x_on_blowup
            else:
                randomize_x_on_blowup = False
        QRS = np.block([[self.Q, self.S], [self.S.T, self.R]])
        wv, self.QRSShaping = genRandomGaussianNoise(N, QRS)
        w = wv[:, : self.state_dim]
        v = wv[:, self.state_dim :]
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = np.zeros((self.state_dim, 1))
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        if w0 is None:
            w0 = np.zeros((self.state_dim, 1))
        if self.input_dim > 0 and u0 is None:
            u0 = np.zeros((self.input_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        for i in tqdm(range(N), "Generating realization"):
            if i == 0:
                Xt_1 = x0
                Wt_1 = w0
                if self.input_dim > 0 and u is not None:
                    Ut_1 = u0
            else:
                Xt_1 = X[(i - 1) : i, :].T
                Wt_1 = w[(i - 1) : i, :].T
                if self.input_dim > 0 and u is not None:
                    Ut_1 = u[(i - 1) : i, :].T
            X[i, :] = (self.apply_param("A", Xt_1) + Wt_1).T
            if u is not None:
                X[i, :] += np.squeeze(self.apply_param("B", Ut_1).T)
            # Check if X[i, :] has blown up
            if (
                np.any(np.isnan(X[i, :]))
                or np.any(np.isinf(X[i, :]))
                or np.any(np.abs(X[i, :]) > blowup_threshold)
            ):
                msg = f"Xp blew up at sample {i} (mean Xp={np.mean(X[i, :]):.3g})"
                if reset_x_on_blowup:
                    X[i, :] = x0
                    msg += f", so it was reset to initial x0 (mean x0={np.mean(X[i, :]):.3g})"
                if randomize_x_on_blowup:
                    X[i, :] = np.atleast_2d(
                        np.random.multivariate_normal(
                            mean=np.zeros(self.state_dim), cov=self.XCov
                        )
                    ).T
                    msg += f", so it was reset to a random Gaussian x0 with XCov (mean x0={np.mean(X[i, :]):.3g})"
                logger.warning(msg)
        Y = v
        CxDu = self.generateObservationFromStates(
            X, u=u, param_names=["C", "D"], prep_model_param="YPrepModel"
        )
        if CxDu is not None:
            Y += CxDu
        out = Y, X
        if return_z:
            Z, ZErr = self.generateZRealizationFromStates(X=X, U=u, return_err=True)
            out += (Z,)
            if return_z_err:
                out += (ZErr,)
        if return_wv:
            out += (wv,)
        return out

    def generateRealizationWithKF(
        self,
        N,
        x0=None,
        u0=None,
        u=None,
        e=None,
        return_z=False,
        return_z_err=False,
        return_e=False,
        blowup_threshold=None,
        reset_x_on_blowup=None,
        randomize_x_on_blowup=None,
    ):
        if blowup_threshold is None:
            if hasattr(self, "blowup_threshold"):
                blowup_threshold = self.blowup_threshold
            else:
                blowup_threshold = np.inf
        if reset_x_on_blowup is None:
            if hasattr(self, "reset_x_on_blowup"):
                reset_x_on_blowup = self.reset_x_on_blowup
            else:
                reset_x_on_blowup = False
        if randomize_x_on_blowup is None:
            if hasattr(self, "randomize_x_on_blowup"):
                randomize_x_on_blowup = self.randomize_x_on_blowup
            else:
                randomize_x_on_blowup = False
        if e is None:
            e, innovShaping = genRandomGaussianNoise(N, self.innovCov)
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = np.zeros((self.state_dim, 1))
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        if self.input_dim > 0 and u0 is None:
            u0 = np.zeros((self.input_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        Xp = x0
        tic = time.perf_counter()
        time_passed = 0
        for i in tqdm(range(N), "Generating realization"):
            ek = e[i, :][:, np.newaxis]
            yk = self.apply_param("C", Xp) + ek
            if u is not None:
                yk += self.apply_param("D", u[i, :][:, np.newaxis])
            X[i, :] = np.squeeze(Xp)
            Y[i, :] = np.squeeze(yk)
            # Xp = self.apply_param('A', Xp) \
            #     - self.apply_param('K', self.apply_param('C', Xp)) \
            #     + self.apply_param('K', yk)
            Xp = self.apply_param("A_KC", Xp) + self.apply_param("K", yk)
            if u is not None:
                Ut = u[i, :][:, np.newaxis]
                Xp += self.apply_param("B_KD", Ut)
            # Check if Xp has blown up
            if (
                np.any(np.isnan(Xp))
                or np.any(np.isinf(Xp))
                or np.any(np.abs(Xp) > blowup_threshold)
            ):
                msg = f"Xp blew up at sample {i} (mean Xp={np.mean(Xp):.3g})"
                if reset_x_on_blowup:
                    Xp = x0
                    msg += (
                        f", so it was reset to initial x0 (mean x0={np.mean(Xp):.3g})"
                    )
                if randomize_x_on_blowup:
                    Xp = np.atleast_2d(
                        np.random.multivariate_normal(
                            mean=np.zeros(self.state_dim), cov=self.XCov
                        )
                    ).T
                    msg += f", so it was reset to a random Gaussian x0 with XCov (mean x0={np.mean(Xp):.3g})"
                logger.warning(msg)
            toc = time.perf_counter()
            print_secs = 60
            if (
                (toc - tic) > print_secs
                and np.mod(toc - tic, print_secs) < 0.5 * print_secs
                and np.mod(time_passed, print_secs) >= 0.5 * print_secs
            ):
                logger.info(
                    "{:.2f}% ({}/{} samples) generated after {:.3g} min(s) and {:.3g} second(s)".format(
                        i / N * 100, i, N, (toc - tic) // 60, (toc - tic) % 60
                    )
                )
            time_passed = toc - tic

        out = Y, X
        if return_z:
            Z, ZErr = self.generateZRealizationFromStates(X=X, U=u, return_err=True)
            out += (Z,)
            if return_z_err:
                out += (ZErr,)
        if return_e:
            out += (e,)
        return out

    def find_fixedpoints(self, Y=None, X=None, U=None, N=None):
        if X is None:
            N = 1000
            Y, X, Z = self.generateRealization(N, return_z=True, u=U)

        if N is None:
            inds = np.arange(X.shape[0])
        else:
            inds = np.arange(np.min((N, X.shape[0])))

        oDiff = (
            X[inds[1:], :].T
            - self.apply_param("K", Y[inds[:-1], :].T)
            - X[inds[:-1], :].T
        ).T
        oDiffNorm = np.sum(oDiff**2, axis=1)
        rootInd = np.argsort(oDiffNorm)
        maxNorm = 1e-2
        rootInd = rootInd[oDiffNorm[rootInd] < maxNorm]
        rootVal = X[rootInd, :]

        if rootVal.size > 0:
            # Subsample to keep at most 10k examples otherwise clustering will be too slow
            if rootVal.size > 1000:
                rootVal = np.random.choice(rootVal.flatten(), 1000, replace=False)[
                    :, np.newaxis
                ]
            from sklearn import cluster

            clustering = cluster.MeanShift().fit(rootVal)
            fpVals = clustering.cluster_centers_
        else:
            fpVals = []
        return fpVals

    def kalman(
        self,
        Y,
        U=None,
        x0=None,
        P0=None,
        steady_state=True,
        blowup_threshold=None,
        clip_on_blowup=None,
        reset_x_on_blowup=None,
        randomize_x_on_blowup=None,
    ):
        if blowup_threshold is None:
            if hasattr(self, "blowup_threshold"):
                blowup_threshold = self.blowup_threshold
            else:
                blowup_threshold = np.inf
        if clip_on_blowup is None:
            if hasattr(self, "clip_on_blowup"):
                clip_on_blowup = self.clip_on_blowup
            else:
                clip_on_blowup = False
        if reset_x_on_blowup is None:
            if hasattr(self, "reset_x_on_blowup"):
                reset_x_on_blowup = self.reset_x_on_blowup
            else:
                reset_x_on_blowup = False
        if randomize_x_on_blowup is None:
            if hasattr(self, "randomize_x_on_blowup"):
                randomize_x_on_blowup = self.randomize_x_on_blowup
            else:
                randomize_x_on_blowup = False
        if self.state_dim == 0:
            allXp = np.zeros((Y.shape[0], self.state_dim))
            allX = allXp
            allYp = np.zeros((Y.shape[0], self.output_dim))
            return allXp, allYp, allX
        if not steady_state:
            raise (Exception("Not supported!"))
        N = Y.shape[0]
        allXp = np.empty((N, self.state_dim))  # X(i|i-1)
        # allX = np.empty((N, self.state_dim))
        allX = None
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = np.zeros((self.state_dim, 1))
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        if P0 is None:
            if hasattr(self, "P0"):
                P0 = self.P0
            else:
                P0 = np.eye(self.state_dim)
        Xp = x0
        Pp = P0
        for i in tqdm(range(N), "Estimating latent states"):
            allXp[i, :] = np.transpose(Xp)  # X(i|i-1)
            thisY = Y[i, :][np.newaxis, :]
            if hasattr(self, "YPrepModel") and self.YPrepModel is not None:
                thisY = self.YPrepModel.apply(
                    thisY, time_first=True
                )  # Apply any mean removal/zscoring

            if U is not None:
                ui = U[i, :][:, np.newaxis]
                if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                    ui = self.UPrepModel.apply(
                        ui, time_first=False
                    )  # Apply any mean removal/zscoring

            if self.missing_marker is not None and np.any(
                Y[i, :] == self.missing_marker
            ):
                newXp = self.apply_param("A", Xp)
                if U is not None and self.B.size > 0:
                    newXp += self.apply_param("B", ui)
            else:
                newXp = self.apply_param("A_KC", Xp) + self.apply_param("K", thisY.T)
                if U is not None:
                    newXp += self.apply_param("B_KD", ui)
            # Check if Xp has blown up
            if (
                np.any(np.isnan(newXp))
                or np.any(np.isinf(newXp))
                or np.any(np.abs(newXp) > blowup_threshold)
            ):
                msg = f"Xp blew up at sample {i} (mean Xp={np.mean(newXp):.3g})"
                if clip_on_blowup:
                    msg += f", so it was clipped to its previous value (mean x0={np.mean(Xp):.3g})"
                    newXp = Xp
                if reset_x_on_blowup:
                    newXp = x0
                    msg += f", so it was reset to initial x0 (mean x0={np.mean(newXp):.3g})"
                if randomize_x_on_blowup:
                    newXp = np.atleast_2d(
                        np.random.multivariate_normal(
                            mean=np.zeros(self.state_dim), cov=self.XCov
                        )
                    ).T
                    msg += f", so it was reset to a random Gaussian x0 with XCov (mean x0={np.mean(newXp):.3g})"
                logger.warning(msg)
            Xp = newXp

        allYp = self.generateObservationFromStates(
            allXp,
            u=U,
            param_names=["C", "D"],
            prep_model_param="YPrepModel",
            mapping_param="cMapY",
        )
        return allXp, allYp, allX

    def propagateStates(self, allXp, step_ahead=1):
        for step in range(step_ahead - 1):
            if (
                hasattr(self, "multi_step_with_A_KC") and self.multi_step_with_A_KC
            ):  # If true, forward predictions will be done with A-KC rather than the correct A (but will be useful for comparing with predictor form models)
                # allXp = self.apply_param('A', allXp.T).T - self.apply_param('K', self.apply_param('C', allXp.T)).T
                allXp = self.apply_param("A_KC", allXp.T).T
            else:
                allXp = self.apply_param("A", allXp.T).T
        return allXp
