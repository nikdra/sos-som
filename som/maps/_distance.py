"""
This module gathers distance functions for weight vectors and data samples
"""

# Authors: Nikola Dragovic (@nikdra), 18.07.2020

import numpy as np


def _euclid_distance(codebook, sample):
    """
    Vectorized computation of the euclidean distance between unit weight vectors and a given sample.

    Parameters
    ----------
    codebook: array-like of shape (map_size, n_features)
        The SOM codebook.
    sample:
        A sample vector from the data. Must have same shape as the last feature dimension of the codebook.

    Returns
    -------
    distance_matrix: array-like of shape map_size
        The euclidean distances for each unit weight vector in the codebook.
    """
    return np.sqrt(np.sum(np.square(codebook - sample), axis=2))
