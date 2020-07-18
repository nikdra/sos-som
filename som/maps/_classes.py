"""
This module gathers SOM variants that can be trained in this module.
"""

# Authors: Nikola Dragovic (@nikdra), 18.07.2020
# TODO add exceptions for faulty parameters

from abc import abstractmethod
import numpy as np

from ._codebook import _init_codebook
from ._distance import _euclid_distance
from ._neighborhood import _generate_neighborhood_indices_2d, _gauss_neighborhood_2d


class BaseSOM:
    """
    Base class for self-organizing maps.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 neighborhood_radius,
                 neighborhood_type,
                 distance_measure):
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius
        self.distance_measure = distance_measure
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
    map_size: int, int
        The size of the rectangular SOM (height, width).
    neighborhood_radius: int
        The radius of the neighborhood.
        For the Gaussian neighborhood, this is the standard deviation of the Gauss function.
    neighborhood_type: {"gauss"}, default = "gauss"
        The type of neighborhood to be used for training the SOM.
    distance_measure: {"euclidean"}, default = "euclidean"
        The distance measure to be used to calculate distances between units' weight vectors and the data

    Attributes
    ----------
    map_size: int, int
        The height and width of the RectangularSOM.
    neighborhood_indices: ndarray of shape (map_size, 2)
        A ndarray that contains the indices of each position the SOM. Needed for vectorization of the update function.
    neighborhood_function: function(ndarray, int, float)
        The neighborhood function to be used in an iteration of the training. The first argument are the SOM indices,
        the second argument is the index of the BMU of an iteration, and the third argument is the current
        neighborhood radius.
    distance_function: function(ndarray, array-like)
        The function for calculating the distances between every weight vector in the codebook and a sample (vector).
    """

    def __init__(self,
                 map_size,
                 neighborhood_radius,
                 neighborhood_type="gauss",
                 distance_measure="euclidean"):
        super().__init__(neighborhood_radius,
                         neighborhood_type,
                         distance_measure)
        # set map size
        self.map_size = map_size
        # set array of neighborhood indices
        self.neighborhood_indices = _generate_neighborhood_indices_2d(map_size)
        # set neighborhood function
        self.neighborhood_function = self.__neighborhood()
        # set distance function
        self.distance_function = self.__distance()

    def train(self, data, iterations=10000, alpha=0.95, random_seed=1, codebook=None):
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

        :math:`m_i \\leftarrow m_i(t) + \\alpha(t) \\cdot h_{ci}(t) \\cdot |x(t) - m_i(t)|`

        The main loop could not be eliminated via vectorization.
        """

        # set random seed
        np.random.seed(random_seed)

        # no custom initialization of the codebook given
        if codebook is None:
            self.codebook = _init_codebook(self.map_size, data.to_numpy())

        # initialize arrays of alphas and radii - decrease linearly with increasing iterations
        alphas = np.linspace(alpha, 0, num=iterations, endpoint=False)
        radii = np.linspace(self.neighborhood_radius, 0, num=iterations, endpoint=False)

        # main training loop
        for i in range(iterations):
            # get data point
            x = data.sample().to_numpy()
            # calculate distance
            d = self.distance_function(self.codebook, x)
            # get index of unit with minimum distance
            ind = np.unravel_index(np.argmin(d), d.shape)
            # get neighborhood
            neighborhood = self.neighborhood_function(self.neighborhood_indices, ind, radii[i])
            # update
            self.codebook = self.codebook + alphas[i] * neighborhood[:, :, None] * (x - self.codebook)

        # finished training
        self.trained = True
        return self

    def __neighborhood(self):
        """
        Set the underlying neighborhood function for the given neighborhood type

        Returns
        -------
        neighborhood_function: function(neighborhood_indices, index, radius)
            The neighborhood function.
        """
        if self.neighborhood_type == "gauss":
            return _gauss_neighborhood_2d

    def __distance(self):
        """
        Set the underlying distance function for the given distance measure

        Returns
        -------
        distance_function: function(codebook, sample)
            The distance function.
        """
        if self.distance_measure == "euclidean":
            return _euclid_distance
