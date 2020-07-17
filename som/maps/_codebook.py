"""
This module gathers codebook functions for SOMs
"""

# Authors: Nikola Dragovic (@nikdra), 18.07.2020

import numpy as np


def _init_codebook(map_size, data):
    """
    Initialize the codebook of shape (map_size, n_features) with random values in [min_value, max_value) for each
    feature dimension in data.

    Parameters
    ----------
    map_size: int, int
        The shape of the SOM.
    data: array-like of shape (n_samples, n_features)
        The data that the SOM will be trained on.

    Returns
    -------
    codebook: array-like of shape (map_size, n_features)
        The initialized codebook.
    """
    # turn map size into list
    mpd = list(map_size)

    # append number of features
    mpd.append(data.shape[1])

    # turn back into a tuple
    arr_size = tuple(mpd)

    # initialize the codebook size width x height x n_features with random values in (0,1]
    codebook = np.random.rand(*arr_size)

    # minimums of features
    data_mins = np.min(data, axis=0)

    # maximums of features
    data_maxs = np.max(data, axis=0)

    # get random weight vectors for units in [min, max) of all features
    # for a feature:
    # value = minimum + [0,1) * (max - min)
    # characteristics: minimum <= value <= maximum
    codebook = codebook * (data_maxs - data_mins) + data_mins

    return codebook
