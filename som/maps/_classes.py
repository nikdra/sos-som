"""
This module gathers SOM variants that can be trained in this module.
"""
# Authors: Nikola Dragovic (@nikdra), 27.07.2020

from abc import abstractmethod
import numpy as np
from scipy.spatial import cKDTree

from ._codebook import _init_codebook
from ._distance import _euclid_distance, _hex_distance
from ._neighborhood import _gauss_neighborhood, _positions_array_generic_2d, generate_hex_positions


class BaseSOM:
    """
    Base class for self-organizing maps.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 topology,
                 neighborhood_radius,
                 neighborhood_type,
                 distance_measure):
        # parameter check
        if topology not in ["rectangular", "hexagonal"]:
            raise ValueError("Topology " + str(topology) + " not supported")
        if neighborhood_radius <= 0:
            raise ValueError("Neighborhood radius smaller or equal 0. Must be greater than 0")
        if neighborhood_type not in ["gauss"]:
            raise ValueError("Neighborhood type " + str(neighborhood_type) + " not supported")
        if distance_measure not in ["euclidean"]:
            raise ValueError("Distance measure " + str(distance_measure) + " not supported")

        self.neighborhood_type = neighborhood_type
        self.topology = topology
        self.neighborhood_radius = neighborhood_radius
        self.distance_measure = distance_measure
        self.codebook = None
        self.positions = None
        self.trained = False
        self.bmu_distances = None
        self.bmu_indices = None

    @abstractmethod
    def train(self, data, iterations=10000, alpha=0.95, random_seed=1, codebook=None):
        raise NotImplementedError()

    def get_codebook(self):
        """
        Return the codebook of the SOM.

        The codebook is only returned if the map has been trained

        Returns
        -------
        codebook: ndarray of shape (n_units, n_features) or None in case the SOM is not trained
            The codebook of the SOM i.e., the weight vectors of the units of the SOM.
        """
        if self.trained:
            return self.codebook
        return None

    def get_positions(self):
        """
        Return the positions of the units of the SOM

        The positions are only returned if the SOM has been initialized


        Returns
        -------
        positions: array-like or None in case the SOM has not been initialized
            The position of each unit of the SOM. Can be of various shapes depending on the architecture of the SOM
        """
        return self.positions

    @abstractmethod
    def get_first_bmus(self):
        raise NotImplementedError()

    @abstractmethod
    def get_second_bmus(self):
        raise NotImplementedError()


class StandardSOM(BaseSOM):
    """
    A standard rectangular SOM.

    Parameters
    ----------
    map_size: int, int
        The size of the rectangular SOM (height, width). Both height and width must be greater than zero.
    neighborhood_radius: float
        The radius of the neighborhood. Must be greater than zero.
        For the Gaussian neighborhood, this is the standard deviation of the Gauss function.
    topology: {"rectangular", "hexagonal"}, default = "rectangular"
        The topology of the SOM. Can be rectangular (4 neighbors) or hexagonal (8 neighbors)
    neighborhood_type: {"gauss"}, default = "gauss"
        The type of neighborhood to be used for training the SOM.
    distance_measure: {"euclidean"}, default = "euclidean"
        The distance measure to be used to calculate distances between units' weight vectors and the data.

    Attributes
    ----------
    map_size: int, int
        The height and width of the StandardSOM.
    topology: {"rectangular", "hexagonal"}
        The topology of the StandardSOM. Determines the number of neighbors for a unit. In a rectangular SOM, a unit
        has four neighbors. In a hexagonal SOM, a unit has six neighbors.
    positions: ndarray of shape (n_units, n_dim)
        The array of positions of each unit of the SOM. In a rectangular grid, this array has two-dimensional entries
        at every index. In a hexagonal grid, this array has three-dimensional entries at every index (cube coordinates).
    neighborhood_function: function(ndarray, float)
        The neighborhood function to be used in an iteration of the training. The first argument are the distances to
        the BMU in the SOM. The second argument is the current neighborhood radius.
    input_space_distance: function(ndarray, array-like)
        The function for calculating the distances between every weight vector in the codebook and a sample (vector).
    output_space_distance: function(ndarray, array-like)
        The function for calculating the distances between every unit in the SOM and a given unit.
    trained: bool
        True if the SOM has been trained, False otherwise.
    bmu_distances: ndarray of shape (n_data, 2)
        An array that contains the distances to the two BMU in the SOM for each data point
    bmu_indices: ndarray of shape (n_data, 2)
        An array that contains the indices of the positions of the two BMU in the SOM data point
    """

    def get_first_bmus(self):
        pass

    def get_second_bmus(self):
        pass

    def __init__(self,
                 map_size,
                 neighborhood_radius,
                 topology="rectangular",
                 neighborhood_type="gauss",
                 distance_measure="euclidean"):
        super().__init__(topology,
                         neighborhood_radius,
                         neighborhood_type,
                         distance_measure)
        # parameter check
        if type(map_size) != tuple:
            raise ValueError("map_size is not a tuple")
        if len(map_size) != 2:
            raise ValueError("map_size must be tuple of length 2")
        if map_size[0] <= 0 or map_size[1] <= 0:
            raise ValueError("height and width of StandardSOM must be greater zero")
        if type(map_size[0]) != int or type(map_size[1]) != int:
            raise ValueError("height and width of map must be integers")

        # set map size
        self.map_size = map_size
        # set array of positions
        if self.topology == "rectangular":
            self.positions = _positions_array_generic_2d(map_size)
        elif self.topology == "hexagonal":
            self.positions = generate_hex_positions(map_size)
        # set neighborhood function
        self.neighborhood_function = self.__neighborhood()
        # set distance function in input space
        self.input_space_distance = self.__input_distance()
        # set distance function in output space
        self.output_space_distance = self.__output_distance()

    def train(self, data, iterations=10000, alpha=0.95, random_seed=1, codebook=None):
        """
        Train the standard rectangular SOM using the iterative algorithm.

        Parameters
        ----------
        data: DataFrame of shape (n_samples, n_features)
            Data to train the SOM. Should not contain the class labels for interpretable results.
        iterations: int, default = 10000
            The number of iterations in the algorithm. Must be greater than zero.
        alpha: double, default = 0.95
            The learning parameter. Decreases linearly towards zero with increasing iterations. Must be greater than
            zero.
        random_seed: int, default = 1
            The random seed for the algorithm as well as the initialization of the codebook.
        codebook: DataFrame of shape (n_units, n_features), default = "None"
            The initial codebook for the SOM. If not set, the SOM will be initialized with random values in the range of
            the minimum of a feature value to its maximum.

        Returns
        -------
        self: StandardSOM
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
        # parameter check
        if iterations <= 0:
            raise ValueError("Iterations must be greater 0")
        if alpha <= 0:
            raise ValueError("Learning parameter must be greater 0")
        if data is None:
            raise ValueError("Data is None")

        # set random seed
        np.random.seed(random_seed)

        # no custom initialization of the codebook given
        if codebook is None:
            # initialize codebook with random values
            self.codebook = _init_codebook(self.map_size[0] * self.map_size[1], data.to_numpy())

        # initialize arrays of alphas and radii - decrease linearly with increasing iterations
        alphas = np.linspace(alpha, 0, num=iterations, endpoint=False)
        radii = np.linspace(self.neighborhood_radius, 0, num=iterations, endpoint=False)

        # main training loop
        for i in range(iterations):
            # get data point
            x = data.sample().to_numpy()
            # calculate distance in input space
            d = self.input_space_distance(self.codebook, x)
            # get index of unit with minimum distance
            ind = np.unravel_index(np.argmin(d), d.shape)
            # get position of unit with minimum distance
            bmu = self.positions[ind]
            # get distances of BMU to all units in output space
            neighborhood_distances = self.output_space_distance(self.positions, bmu)
            # get neighborhood
            neighborhood = self.neighborhood_function(neighborhood_distances, radii[i])
            # update
            self.codebook = self.codebook + alphas[i] * neighborhood[:, None] * (x - self.codebook)

        # find the first and second BMU for each data point
        # TODO adapt when other distance measures are implemented
        p = 2
        self.__find_bmu(data, p)

        # finished training
        self.trained = True
        return self

    def __find_bmu(self, data, p):
        """
        Find the first and second BMU for each data point. The result is stored in the RectangularSOM.

        The notion of the BMU itself is also dependent on the output space distance measure (euclidean, minkowski,
        city-block etc.).

        Nearest neighbor search for high dimensions is an open problem in computer science. Search in high-dimensional
        domains is essentially brute-force, but can be assisted by building KD-Trees, which are faster for
        lower-dimensional data. We use the scipy implementation of KD-Trees to find the BMUs.

        Parameters
        ----------
        data: DataFrame of shape (n_samples, n_features)
            Data to train the SOM. Should not contain the class labels for interpretable results.
        p: float, 1 <= p <= infinity
            Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance 2 is the usual Euclidean
            distance infinity is the maximum-coordinate-difference distance A finite large p may cause a ValueError if
            overflow can occur.

        Returns
        -------
        None
        """
        tree = cKDTree(self.codebook)
        self.bmu_distances, self.bmu_indices = tree.query(data, k=2, p=p)

    def __neighborhood(self):
        """
        Set the underlying neighborhood function for the given neighborhood type

        Returns
        -------
        neighborhood_function: function(neighborhood_distances, radius)
            The neighborhood function.
        """
        if self.neighborhood_type == "gauss":
            return _gauss_neighborhood

    def __input_distance(self):
        """
        Set the underlying distance function for the given distance measure in input space

        Returns
        -------
        input_distance_function: function(codebook, sample)
            The input space distance function.
        """
        if self.distance_measure == "euclidean":
            return _euclid_distance

    def __output_distance(self):
        """
        Set the underlying distance function for the given topology in output space


        Returns
        -------
        output_distance_function: function(positions, position)
            The output space distance function.
        """
        if self.topology == "rectangular":
            return _euclid_distance
        elif self.topology == "hexagonal":
            return _hex_distance
