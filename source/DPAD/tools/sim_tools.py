""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Tools for simulating models"""

import numpy as np
from scipy import stats


def addConjs(vals):
    """Adds complex conjugate values for each complex value

    Args:
        vals (list of number): list of numbers (e.g. eigenvalues)

    Returns:
        output (list of numbers): new list of numbers where for each complex value in the original list,
            the conjugate is also added to the new list. For real values an np.nan is added.
    """
    vals = np.atleast_2d(vals).T
    valsConj = vals.conj()
    valsConj[np.abs(vals - valsConj) < np.spacing(1)] = np.nan
    return np.concatenate((vals, valsConj), axis=1)


def drawRandomPoles(N, poleDist={}):
    """Draws random eigenvalues from the unit dist

    Args:
        N (int): number of eigenvalues to draw
        poleDist (dict, optional): information about the distribution.
            For options see the top of the code. Defaults to dict.

    Returns:
        valsA (list of numbers): drawn random values
    """
    nCplx = int(np.floor(N / 2))

    if "magDist" not in poleDist:
        poleDist["magDist"] = "beta"
    if "magDistParams" not in poleDist and poleDist["magDist"] == "beta":
        poleDist["magDistParams"] = {"a": 2, "b": 1}
    if "angleDist" not in poleDist:
        poleDist["angleDist"] = "uniform"

    # mag = np.random.rand(nCplx) # Uniform dist
    if poleDist["magDist"] == "beta":
        a, b = 2, 1  # Use a, b = 2, 1 for uniform prob over unit circle
        if "a" in poleDist["magDistParams"]:
            a = poleDist["magDistParams"]["a"]
        if "b" in poleDist["magDistParams"]:
            b = poleDist["magDistParams"]["b"]

        """
        import matplotlib.pyplot as plt    
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(0, 1, 100)
        ax.plot(x, stats.beta.pdf(x, a, b), lw=5, alpha=0.6, label='beta pdf (a={}, b={})'.format(a, b))
        ax.legend()
        plt.show()
        """

        mag = stats.beta.rvs(a=a, b=b, size=nCplx)  # Beta dist
    else:
        raise Exception("Only beta distribution is supported for the magnitude")

    if poleDist["angleDist"] == "uniform":
        theta = np.random.rand(nCplx) * np.pi
    else:
        raise Exception("Only uniform distribution is supported for the angle")

    vals = mag * np.exp(1j * theta)

    valsA = addConjs(vals)
    valsA = valsA.reshape(valsA.size)
    valsA = valsA[np.logical_not(np.isnan(valsA))]

    # Add real mode(s) if needed
    nReal = N - 2 * nCplx
    if nReal > 0:
        # rVals = np.random.rand(nReal)
        rVals = stats.beta.rvs(a=a, b=b, size=nReal)  # Beta dist
        rSign = 2 * (((np.random.rand(nReal) > 0.5).astype(float)) - 0.5)

        valsA = np.concatenate((valsA, rVals * rSign))

    return valsA
