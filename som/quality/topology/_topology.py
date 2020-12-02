import numpy as np
from scipy.spatial.distance import pdist

from som.maps import BaseSOM
from som.maps._distance import _hex_point_distance


def topographic_error(som: BaseSOM):
    """
    Calculate the topographic error for the SOM.

    The topographic error is the percentage of data points where the :math:`2^{nd}` BMU is not neighbouring the BMU.
    This results in a single number between 0 and 1. Note that we only consider the 4-neighbor variant for SOMs with
    rectangular topologies.

    Parameters
    ----------
    som: BaseSOM
        The trained SOM

    Returns
    -------
    topographic_error: float
        The topographic error between 0 and 1.
    """
    # init variable
    bmu_dist = None
    if som.topology == 'rectangular':
        # calculate euclid. distance between units in output space
        # enter 0 if units are neighboring, else 1
        bmu_dist = [0 if pdist(points)[0] == 1 else 1 for points in som.positions[som.bmu_indices]]
    elif som.topology == 'hexagonal':
        # calculate manhattan distance between units in output space
        # enter 0 if units are neighboring, else 1
        bmu_dist = [0 if _hex_point_distance(*points) == 1 else 1 for points in som.positions[som.bmu_indices]]

    # average of neighboring array = topographic error
    return np.mean(bmu_dist)
