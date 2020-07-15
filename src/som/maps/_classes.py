from abc import abstractmethod


class BaseSOM:
    """Base class for self-organizing maps.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 neighborhood_radius,
                 neighborhood_type,
                 codebook):
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius
        self.codebook = codebook
        self.trained = False

    @abstractmethod
    def train(self, data, iterations=10000, alpha=0.95, random_seed=1):
        pass

    def get_codebook(self):
        """Return the codebook of the SOM.

        The codebook is only returned if the map has been trained
        :return: The codebook of the SOM as a pandas DataFrame
        """
        if self.trained:
            return self.codebook
        return None


class RectangularSOM(BaseSOM):
    """A standard rectangular SOM

    Parameters
    ----------
    neighborhood_radius: int
        The radius of the neighborhood.
        For the Gaussian neighborhood, this is the standard deviation of the Gauss function.

    neighborhood_type: {"gauss"}, default = "gauss"
        The type of neighborhood to be used for training the SOM

    codebook: DataFrame, default = "None"
        The initial codebook for the SOM. If not set, the SOM will be initialized with random values in the range of
        the minimum of a feature value to its maximum.

    map_size: (int, int)
        The size of the rectangular SOM (height, width).
    """

    def __init__(self,
                 map_size,
                 neighborhood_radius,
                 neighborhood_type="gauss",
                 codebook=None):
        super().__init__(neighborhood_radius,
                         neighborhood_type,
                         codebook)
        self.map_size = map_size

    def train(self, data, iterations=10000, alpha=0.95, random_seed=1):
        """Train the standard rectangular SOM using the iterative algorithm

        :param data: DataFrame
            Data to train the SOM. Should not contain the class labels for interpretable results.
        :param iterations: int, default = 10000
            The number of iterations in the algorithm.
        :param alpha: double, default = 0.95
            The learning parameter. Decreases linearly towards zero with increasing iterations.
        :param random_seed: int, default = 1
            The random seed for the algorithm
        :return: self: RectangularSOM
            Fitted SOM
        """
        self.trained = True
        return self
