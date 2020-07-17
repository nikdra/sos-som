"""
This module gathers SOM variants that can be trained in this module.
"""

# Authors: Nikola Dragovic (@nikdra), 18.07.2020

from abc import abstractmethod
import numpy as np
from scipy.stats import multivariate_normal

from ._codebook import _init_codebook
from ._distance import _euclid_distance
from ._neighborhood import _generate_neighborhood_indices_2d, _gauss_neighborhood


class BaseSOM:
    """
    Base class for self-organizing maps.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 neighborhood_radius,
                 neighborhood_type):
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius
        self.codebook = None
        self.trained = False

    @abstractmethod
    def train(self, data, iterations=10000, alpha=0.95, random_seed=1, codebook=None):
        pass

    def get_codebook(self):
        """
        Return the codebook of the SOM.

        The codebook is only returned if the map has been trained

        Returns
        -------
        codebook: ndarray of shape (map_dim, n_features) or None in case the SOM is not trained
            The codebook of the SOM i.e. the weight vectors of the units of the SOM.
        """
        if self.trained:
            return self.codebook
        return None


class RectangularSOM(BaseSOM):
    """
    A standard rectangular SOM.

    Parameters
    ----------
    neighborhood_radius: int
        The radius of the neighborhood.
        For the Gaussian neighborhood, this is the standard deviation of the Gauss function.

    neighborhood_type: {"gauss"}, default = "gauss"
        The type of neighborhood to be used for training the SOM.

    map_size: int, int
        The size of the rectangular SOM (height, width).
    """

    def __init__(self,
                 map_size,
                 neighborhood_radius,
                 neighborhood_type="gauss"):
        super().__init__(neighborhood_radius,
                         neighborhood_type)
        self.map_size = map_size
        self.neighborhood_indices = _generate_neighborhood_indices_2d(map_size)

    def train(self, data, iterations=10000, alpha=0.95, random_seed=1, codebook=None):
        # TODO check neighborhood function, distance function (add also as parameter)
        """
        Train the standard rectangular SOM using the iterative algorithm.

        Parameters
        ----------
        data: DataFrame of shape (n_samples, n_features)
            Data to train the SOM. Should not contain the class labels for interpretable results.
        iterations: int, default = 10000
            The number of iterations in the algorithm.
        alpha: double, default = 0.95
            The learning parameter. Decreases linearly towards zero with increasing iterations.
        random_seed: int, default = 1
            The random seed for the algorithm as well as the initialization of the codebook.
        codebook: DataFrame of shape (map_size, n_features), default = "None"
            The initial codebook for the SOM. If not set, the SOM will be initialized with random values in the range of
            the minimum of a feature value to its maximum.

        Returns
        -------
        self: RectangularSOM
            Fitted SOM

        Notes
        -----
        The neighborhood radius of the neighborhood function (in case of "gauss") as well as the
        learning parameter decrease linearly with increasing number of iterations. The implementation here is the
        algorithm presented in the Self-Organizing Systems lecture, as also described in Kohonen (1982).
        The update rule is implemented in a vectorized fashion for the codebook, but follows the known schema:

        :math:`m_i \\leftarrow m_i(t) + \\alpha(t) \\cdot h_{ci}(t) \\cdot |x_i(t) - m_i(t)|`

        The main loop could not be eliminated via vectorization.
        """

        # set random seed
        np.random.seed(random_seed)

        if codebook is None:
            self.codebook = _init_codebook(self.map_size, data.to_numpy())

        # initialize arrays of alphas and radii - decrease linearly with increasing iterations
        alphas = np.linspace(alpha, 0, num=iterations)
        radii = np.linspace(self.neighborhood_radius, 0, num=iterations)

        # main training loop
        for i in range(iterations):
            # get data point
            x = data.sample().numpy()
            # calculate distance
            d = _euclid_distance(codebook, x)
            # get index of unit with minimum distance
            ind = np.unravel_index(np.argmin(d), d.shape)
            # get neighborhood
            neighborhood = _gauss_neighborhood(self.neighborhood_indices, ind, radii[i])
            # update
            codebook = codebook + alphas[i] * neighborhood[:, :, None] * (x - codebook)

        # finished training
        self.trained = True
        return self
