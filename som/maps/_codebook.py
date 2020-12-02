"""
This module gathers codebook functions for SOMs
"""

# Authors: Nikola Dragovic (@nikdra), 26.07.2020

import numpy as np


def _init_codebook(n_units, data):
    """
    Initialize the codebook of shape (n_units, n_features) with random values in [min_value, max_value) for each
    feature dimension in data.

    Parameters
    ----------
    n_units: int
        The number of units in the SOM.
    data: array-like of shape (n_samples, n_features)
        The data that the SOM will be trained on.

    Returns
    -------
    codebook: array-like of shape (n_units, n_features)
        The initialized codebook.
    """

    # initialize the codebook size n_units x n_features with random values in (0,1]
    codebook = np.random.rand(n_units, data.shape[1])

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
