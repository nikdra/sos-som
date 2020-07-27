import numpy as np

from som.maps import BaseSOM


def qe_m(som: BaseSOM):
    """
    Calculate the quantization error for all units in the SOM.

    The quantization error for a unit is defined as

    :math:`qe(m) = \\sum_{x \\in W_m} \\Vert x - m \\Vert`

    Parameters
    ----------
    som: BaseSOM
        The trained SOM.

    Returns
    -------
    qe_m: array of size n_units
        The quantization error for each unit.
    """
    # initialize result array
    res = np.zeros(som.get_codebook().shape[0])

    # aggregate
    agg = {k: np.sum(v) for k, v in som.get_first_bmus().items()}

    # fill up values at the resp. positions
    res[list(agg.keys())] = list(agg.values())

    # return result
    return res


def mqe_m(som: BaseSOM):
    """
    Calculate the mean quantization error for all units in the SOM.

    The mean quantization error for a unit is defined as

    :math:`mqe(m) = \\frac{1}{W_m} \\cdot \\sum_{x \\in W_m} \\Vert x - m \\Vert`

    Parameters
    ----------
    som: BaseSOM
        The trained SOM.

    Returns
    -------
    mqe_m: array of size n_units
        The mean quantization error for each unit.
    """
    # initialize result array
    res = np.zeros(som.get_codebook().shape[0])

    # aggregate
    agg = {k: np.mean(v) for k, v in som.get_first_bmus().items()}

    # fill up values at the resp. positions
    res[list(agg.keys())] = list(agg.values())

    # return result
    return res


def qe(som: BaseSOM):
    """
    Calculate the map quantization error for the SOM.

    The map quantization error for a unit is defined as

    :math:`QE = \\sum_{m \\in M} \sum_{x \\in W_m} \\Vert x - m \\Vert = \\sum_{m \\in M} qe(m)`

    We can compute the :math:`QE` as the sum of :math:`qe(m)`

    Parameters
    ----------
    som: BaseSOM
        The trained SOM.

    Returns
    -------
    qe: float
        The map quantization error for the SOM.
    """
    return np.sum(qe_m(som))


def mqe(som: BaseSOM):
    """
    Calculate the mean map quantization error for the SOM.

    The mean map quantization error for a unit is defined as

    :math:`mQE = \\frac{1}{\\vert M \\vert} \\cdot \\sum_{m \\in M} \\sum_{x \\in W_m} \\Vert x - m \\Vert =
    \\frac{1}{\\vert M \\vert} \\cdot QE`

    We can compute the :math:`mQE` with the number of units and the :math:`QE`

    Parameters
    ----------
    som: BaseSOM
        The trained SOM.

    Returns
    -------
    mqe: float
        The mean map quantization error for the SOM.
    """
    return 1/(som.get_codebook().shape[0]) * qe(som)


def mmqe(som: BaseSOM):
    """
    Calculate the mean mean map quantization error for the SOM.

    The mean mean map quantization error for a unit is defined as

    :math:`mmQE = \\frac{1}{\\vert M \\vert} \\cdot \\sum_{m \\in M} mqe(m)`

    This can be computed as the mean of mean map quantization errors of the units in a SOM.

    Parameters
    ----------
    som: BaseSOM
        The trained SOM.

    Returns
    -------
    mmqe: float
        The mean mean map quantization error for the SOM.
    """
    return np.mean(mqe_m(som))