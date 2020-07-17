"""
This module gathers neighborhood-related functions for SOM training
"""

# Authors: Nikola Dragovic (@nikdra), 18.07.2020

import numpy as np
from scipy.stats import multivariate_normal


def _indices_array_generic_2d(m, n):
    """
    Helper function to generate indices of shape (m, n).

    Parameters
    ----------
    m: int
        Size of the first dimension.
    n: int
        Size of the second dimension.

    Returns
    -------
    out: ndarray of shape (m, n, 2)
        Contains the index (i, j) at every position.
    """
    r0 = np.arange(m)  # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n)
    out = np.empty((m, n, 2), dtype=int)
    out[:, :, 0] = r0[:, None]
    out[:, :, 1] = r1
    return out


def _generate_neighborhood_indices_2d(map_size):
    """
    Generate a ndarray of shape map_size that contains the indices of each index.

    Parameters
    ----------
    map_size: int, int
        The shape of the SOM.

    Returns
    -------
    neighborhood_indices: ndarray of shape (map_size, map_dim)
        A ndarray that contains the indices of each position the SOM.

    Notes
    -----
    This _should_ improve the efficiency of neighborhood calculation via vectorization.
    """
    return _indices_array_generic_2d(*map_size)


def _gauss_neighborhood(neighborhood_indices, mean, sigma):
    """
    Generate the normalized [0,1] pdf of a Gauss distribution for a given neighborhood

    Parameters
    ----------
    neighborhood_indices: ndarray of shape (map_size, map_dim)
        A ndarray that contains the indices of each position the SOM.
    mean: {array-like, tuple} int
        The mean of the Gauss distribution. Usually, this is the position (index) of the BMU
    sigma: float
        The standard deviation of the Gauss distribution. Akin to the neighborhood radius.

    Returns
    -------
    neighborhood: ndarray of shape map_size
        An array that contains normalized values in [0,1] that indicate how much each unit should be pulled
        towards the data sample.
    """
    # calculate pdf with mean at BMU
    sigma = np.array([sigma, sigma])
    neighborhood = multivariate_normal.pdf(neighborhood_indices, mean=mean, cov=np.diag(sigma ** 2))
    # normalize pdf values in [0, 1]
    return (neighborhood - np.min(neighborhood)) / np.ptp(neighborhood)
