import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import filtfilt, lfilter
from scipy.stats import norm


class GaussianSmoother():
    def __init__(self, std, Fs=1, axis=0, kernel_multiplier=None, causal=False):
        """_summary_

        Args:
            std (float): standard deviation of the gaussian kernel to be used. In seconds. If Fs not provided, std 
                can also be though of as being in the unit of signal samples.
            Fs (int, optional): sampling rate of signal. Defaults to 1.
        """        
        self.std = std # In seconds
        self.Fs = Fs
        self.kernel_sigma = std * Fs # effective std for the gaussian kernel

        self.axis = axis
        if kernel_multiplier is None:
            conf = 0.99
            kernel_multiplier = norm.ppf( 1 - 0.5 * (1-conf) )
        self.kernel_multiplier = kernel_multiplier
        self.causal = causal

        if self.causal:
            sigma = self.kernel_sigma
        else:
            sigma = self.kernel_sigma / np.sqrt(2) # Because applying the smoothing filter twice with filtfilt multiplies its effective std by sqrt(2)

        m = self.kernel_multiplier
        filterLen = int(sigma*m + 0.5)
        x0 = 0
        xVals = np.arange(-filterLen, filterLen+1, 1)
        weights = np.exp( -0.5 * (xVals-x0)**2 / sigma**2 )
        if self.causal:
            w_norm = np.sum(weights)
        else:
            w_padded = np.concatenate((np.zeros_like(weights), weights, np.zeros_like(weights)))
            w_norm = np.sqrt( np.sum(convolve1d(w_padded, w_padded, mode='constant', cval=0, origin=0)) )

        weights = weights / w_norm

        self.weights = weights

    def apply(self, data, axis=None):
        if axis is None:
            axis = self.axis
        if self.causal:
            filtered_data = lfilter(self.weights, 1, data, axis=axis)
        else:
            filtered_data = filtfilt(self.weights, 1, data, axis=axis, method='pad', padtype='constant')
        return filtered_data