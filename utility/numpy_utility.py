import numpy as np


def topk(input, k, axis=None, ascending=True):
    """
    Taken from https://hippocampus-garden.com/numpy_topk/
    """

    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis)
    return ind, val
