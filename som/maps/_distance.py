"""
This module gathers distance functions
"""

# Authors: Nikola Dragovic (@nikdra), 30.07.2020

import numpy as np


def _euclid_distance(matrix, vector):
    """
    Vectorized computation of the euclidean distance between a matrix and a vector.

    Parameters
    ----------
    matrix: array-like of shape (n, m)
        A matrix with n rows and m columns.
    vector:
        A vector. Must have size m.

    Returns
    -------
    distance_matrix: array-like of size n
        The euclidean distances for each entry in the matrix and the vector.
    """
    return np.sqrt(np.sum(np.square(matrix - vector), axis=1))


def _hex_distance(hex_positions, hex_position):
    """
    Vectorized computation of the manhattan distance between a matrix and a vector with cube coordinates.

    Parameters
    ----------
    hex_positions: array-like of shape (n, 3)
        A matrix with n rows of cube coordinate positions.
    hex_position: array of size 3
        A cube coordinate. Must have size 3.

    Returns
    -------
    distance_matrix: array-like of size n
        The euclidean distances for each entry in the matrix and the vector.
    """
    return np.sum(np.abs(hex_positions - hex_position), axis=1)/2


def _hex_point_distance(hex_position1, hex_position2):
    """
    Vectorized computation of the manhattan distance between two vectors with cube coordinates.

    Parameters
    ----------
    hex_position1: array of size 3
        A cube coordinate. Must have size 3.
    hex_position2: array of size 3
        A cube coordinate. Must have size 3.

    Returns
    -------
    distance: int
        The distance between two cube coordinates.
    """
    return np.sum(np.abs(hex_position1 - hex_position2))/2
