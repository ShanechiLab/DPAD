""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Tensorflow losses"""

import tensorflow as tf


def masked_mse(mask_value=None):
    """Returns a tf MSE loss computation function, but with support for setting one value as a mask indicator

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(y_true, y_pred):
        # mse = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1) # Without handling NaNs
        # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
        sh = tf.shape(y_true)
        y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
        y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
        y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
        y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)
        if mask_value is not None:
            mask_value_cast = tf.constant(mask_value, dtype=y_true_f.dtype)
            isOk = tf.not_equal(y_true_f, mask_value_cast)
        else:
            isOk = tf.ones_like(y_true_f, dtype=bool)
        isOk1 = tf.math.reduce_all(isOk, axis=-1)
        y_true_masked = tf.boolean_mask(y_true_f, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(y_pred_f, isOk1, axis=0)
        lossFunc = tf.keras.losses.MeanSquaredError()
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = str("MSE_maskV_{}".format(mask_value))
    return f


def compute_CC(x, y):  # https://stackoverflow.com/a/58890795/2275605
    """Computes correlation coefficient (CC) in tensorflow

    Args:
        x (numpy array): input 1
        y (numpy array): input 2

    Returns:
        tf.Tensor: CC value
    """
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


def compute_R2(y_true, y_pred):  # https://stackoverflow.com/a/58890795/2275605
    """Computes correlation of determination (R2) in tensorflow

    Args:
        x (numpy array): input 1
        y (numpy array): input 2

    Returns:
        tf.Tensor: CC value
    """
    m_true = tf.math.reduce_mean(y_true, axis=0)

    r_num = tf.math.reduce_sum(tf.math.pow(y_true - y_pred, 2), axis=0)
    r_den = tf.math.reduce_sum(tf.math.pow(y_true - m_true, 2), axis=0)

    R2 = 1 - (r_num / r_den)

    isFlat = (tf.reduce_max(y_true, axis=0) - tf.reduce_min(y_pred, axis=0)) < 1e-9
    R2 = tf.where(isFlat, tf.zeros_like(R2), R2)
    return R2


def computeCC_masked(y_true, y_pred, mask_value=None):
    """Computes correlation coefficient (CC) in tensorflow, with support for a masked value.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the CC computation.
    Args:
        y_true (numpy array): input 1.
        y_pred (numpy array): input 2
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.

    Returns:
        tf.Tensor: CC value
    """
    # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
    sh = tf.shape(y_true)
    y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
    y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)
    if mask_value is not None:
        mask_value_cast = tf.constant(mask_value, dtype=y_true_f.dtype)
        isOk = tf.not_equal(y_true_f, mask_value_cast)
    else:
        isOk = tf.ones_like(y_true_f, dtype=bool)
    isOk1 = tf.math.reduce_all(isOk, axis=-1)
    y_true_masked = tf.boolean_mask(y_true_f, isOk1, axis=0)
    y_pred_masked = tf.boolean_mask(y_pred_f, isOk1, axis=0)
    CC = compute_CC(y_true_masked, y_pred_masked)
    return CC


def computeR2_masked(y_true, y_pred, mask_value=None):
    """Computes correlation of determination (R2) in tensorflow, with support for a masked value.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the CC computation.
    Args:
        y_true (numpy array): input 1.
        y_pred (numpy array): input 2
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.

    Returns:
        tf.Tensor: R2 value
    """
    # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
    sh = tf.shape(y_true)
    y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
    y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)
    if mask_value is not None:
        mask_value_cast = tf.constant(mask_value, dtype=y_true_f.dtype)
        isOk = tf.not_equal(y_true_f, mask_value_cast)
    else:
        isOk = tf.ones_like(y_true_f, dtype=bool)
    isOk1 = tf.math.reduce_all(isOk, axis=-1)
    y_true_masked = tf.boolean_mask(y_true_f, isOk1, axis=0)
    y_pred_masked = tf.boolean_mask(y_pred_f, isOk1, axis=0)
    R2 = compute_R2(y_true_masked, y_pred_masked)
    return R2


def masked_CC(mask_value=None):
    """Returns a tf correlation coefficient (CC) computation function, but with support for setting one value as a mask indicator.
    Takes mean of CC across dimensions. See computeCC_masked for details of computing CC for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        meanCC = tf.math.reduce_mean(
            computeCC_masked(y_true, y_pred, mask_value)
        )  # Average across dimensions
        return meanCC

    f.__name__ = str("CC_maskV_{}".format(mask_value))
    return f


def masked_R2(mask_value=None):
    """Returns a tf R2 computation function, but with support for setting one value as a mask indicator.
    Takes mean of R2 across dimensions. See computeR2_masked for details of computing R2 for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        allR2 = computeR2_masked(y_true, y_pred, mask_value)
        meanR2 = tf.math.reduce_mean(allR2)  # Average across dimensions
        return meanR2

    f.__name__ = str("R2_maskV_{}".format(mask_value))
    return f


def masked_negativeCC(mask_value=None):
    """Returns a tf negative correlation coefficient (CC) computation function, but with support for setting one value as a mask indicator.
    Takes mean of negative CC across dimensions. See computeCC_masked for details of computing CC for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        meanCC = tf.math.reduce_mean(
            computeCC_masked(y_true, y_pred, mask_value)
        )  # Average across dimensions
        return -meanCC

    f.__name__ = str("negCC_maskV_{}".format(mask_value))
    return f


def masked_negativeR2(mask_value=None):
    """Returns a tf negative correlation of determination (R2) computation function, but with support for setting one value as a mask indicator.
    Takes mean of negative R2 across dimensions. See computeR2_masked for details of computing R2 for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        meanR2 = tf.math.reduce_mean(
            computeR2_masked(y_true, y_pred, mask_value)
        )  # Average across dimensions
        return -meanR2

    f.__name__ = str("negR2_maskV_{}".format(mask_value))
    return f


def masked_PoissonLL_loss(mask_value=None):
    """Returns a tf function that computes the poisson negative log likelihood loss, with support for setting one value as a mask indicator.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the loss computation.

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(true_counts, pred_logLambda):
        sh = tf.shape(true_counts)
        true_counts_f = tf.reshape(true_counts, [tf.reduce_prod(sh[:-1]), sh[-1]])
        pred_logLambda_f = tf.reshape(pred_logLambda, [tf.reduce_prod(sh[:-1]), sh[-1]])
        if mask_value is not None:
            mask_value_cast = tf.constant(int(mask_value), dtype=true_counts_f.dtype)
            isOk = tf.not_equal(true_counts_f, mask_value_cast)
        else:
            isOk = tf.ones_like(true_counts_f, dtype=bool)
        isOk1 = tf.math.reduce_all(isOk, axis=-1)
        y_true_masked = tf.boolean_mask(true_counts_f, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(pred_logLambda_f, isOk1, axis=0)
        # LL = true_counts_f * pred_logLambda_f - tf.math.exp(pred_logLambda_f) - tf.math.lgamma( true_counts_f+1 )
        # pLoss = - tf.reduce_mean(tf.boolean_mask(LL, isOk))
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/poisson
        lossFunc = tf.keras.losses.Poisson()
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = str("PoissonLL_maskV_{}".format(mask_value))
    return f


def masked_CategoricalCrossentropy(mask_value=None):
    """Returns a tf function that computes the Categorical Crossentropy loss, but with support for setting one value as a mask indicator.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the loss computation.

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(y_true, y_pred):
        # Assumes that the last two dimensions are the only data dimensions, others are sample dimensions (batch,time,etc)
        sh = tf.shape(y_true)
        y_true = tf.reshape(y_true, [tf.reduce_prod(sh[:-2]), sh[-2], sh[-1]])
        y_pred = tf.reshape(y_pred, [tf.reduce_prod(sh[:-2]), sh[-2], sh[-1]])
        if mask_value is not None:
            mask_value_cast = tf.constant(int(mask_value), dtype=y_true.dtype)
            isOk = tf.not_equal(y_true, mask_value_cast)
        else:
            isOk = tf.ones_like(y_true, dtype=bool)
        isOk1 = tf.math.reduce_all(
            isOk, axis=tf.range(tf.rank(isOk) - 2, tf.rank(isOk))
        )
        y_true_masked = tf.boolean_mask(y_true, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(y_pred, isOk1, axis=0)
        lossFunc = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )  # Later will need softmax for pred_model
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = str("CCE_maskV_{}".format(mask_value))
    return f


def masked_SparseCategoricalCrossentropy(mask_value=None):
    """Returns a tf function that computes the Sparse Categorical Crossentropy loss, but with support for setting one value as a mask indicator.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the loss computation.

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(y_true, y_pred):
        # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
        sh = tf.shape(y_true)
        y_true = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
        sh2 = tf.shape(y_pred)
        y_pred = tf.reshape(y_pred, [tf.reduce_prod(sh2[:-2]), sh2[-2], sh2[-1]])
        if mask_value is not None:
            mask_value_cast = tf.constant(int(mask_value), dtype=y_true.dtype)
            isOk = tf.not_equal(y_true, mask_value_cast)
        else:
            isOk = tf.ones_like(y_true, dtype=bool)
        isOk1 = tf.math.reduce_all(isOk, axis=-1)
        y_true_masked = tf.boolean_mask(y_true, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(y_pred, isOk1, axis=0)
        lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )  # Later will need softmax for pred_model
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = str("SCCE_maskV_{}".format(mask_value))
    return f
