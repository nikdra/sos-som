"""
This module gathers neighborhood-related functions for SOM training
"""

# Authors: Nikola Dragovic (@nikdra), 26.07.2020

import numpy as np
from scipy.stats import multivariate_normal


def _positions_array_generic_2d(map_size):
    """
    Helper function to generate indices of shape (m * n, 2).

    Parameters
    ----------
    map_size: int, int
         The height and width of the grid.

    Returns
    -------
    out: ndarray of shape (m * n, 2)
        Contains the index [i, j] of each unit.
    """
    m = map_size[0]
    n = map_size[1]
    return np.indices((m, n)).transpose(1, 2, 0).reshape(-1, 2)


def generate_hex_positions(map_size):
    """
    Helper function to generate cube coordinates for a hexagonal grid of rectangular shape

    Parameters
    ----------
    map_size: int, int
         The height and width of the grid.

    Returns
    -------
    arr: ndarray of shape (n_units, 3)
        Contains the cube coordinates of each unit of the grid
    """
    i = 0  # position count
    arr = np.empty((map_size[0] * map_size[1], 3))  # init array of positions (num_units, 3)
    for q in range(map_size[1]):
        q_offset = q // 2  # akin to floor(q/2)
        for r in range(-q_offset, map_size[0] - q_offset):
            arr[i] = np.array([q, r, -q-r])  # add position to array
            i = i + 1  # increase position counter
    return arr


def _gauss_neighborhood(neighborhood_distances, sigma):
    """
    Generate the normalized [0,1] pdf of a Gauss distribution for a given 1d neighborhood

    Parameters
    ----------
    neighborhood_distances: ndarray of size n_units
        A ndarray that contains the distance of each unit of the SOM to the mean.
    sigma: float
        The standard deviation of the Gauss distribution. Akin to the neighborhood radius.

    Returns
    -------
    neighborhood: ndarray of size n_units
        An array that contains normalized values in [0,1] that indicate how much each unit should be pulled
        towards the data sample.
    """
    neighborhood = multivariate_normal.pdf(neighborhood_distances, mean=0, cov=sigma ** 2)
    return __norm_neighborhood(neighborhood)


def __norm_neighborhood(neighborhood):
    """
    Normalize the values of the neighborhood between [0,1]

    Parameters
    ----------
    neighborhood: array of size n_units
        The calculated neighborhood values for each unit in the SOM

    Returns
    -------
    norm_neighborhood:
    """
    return (neighborhood - np.min(neighborhood)) / np.ptp(neighborhood)
