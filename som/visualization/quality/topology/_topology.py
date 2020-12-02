import numpy as np

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from som.maps import StandardSOM, BaseSOM
from som.maps._distance import _hex_point_distance
from som._util.util import group_by


def __standard_som_topographic_error(som: StandardSOM, cmap: str):
    """
    Plot the topographic error for each unit for a standard SOM.
    Plot depends on the topology of the neighborhood (hexagonal or rectangular).

    Parameters:
    -----------
    som: StandardSOM
        The SOM for which the topographic error should be plotted.
    cmap: str
        The matplotlib color map for the map.

    Returns:
    --------
    None
    """
    if som.codebook is not None and som.trained:
        # init variable
        bmu_dist = None
        # TODO refactor with som.quality.topology
        if som.topology == 'rectangular':
            # calculate euclid. distance between units in output space
            # enter 0 if units are neighboring, else 1
            bmu_dist = [0 if pdist(points)[0] == 1 else 1 for points in som.positions[som.bmu_indices]]
        elif som.topology == 'hexagonal':
            # calculate manhattan distance between units in output space
            # enter 0 if units are neighboring, else 1
            bmu_dist = [0 if _hex_point_distance(*points) == 1 else 1 for points in som.positions[som.bmu_indices]]

        # zip errors with bmu indices
        errors = zip(som.bmu_indices[:, 0], bmu_dist)

        # group by index
        grouped = group_by(lambda pair: pair[0], errors)

        # aggregate
        agg = {k: np.sum(v) for k, v in grouped.items()}

        # initialize result array
        res = np.zeros(som.codebook.shape[0])

        # fill up values at the resp. positions
        res[list(agg.keys())] = list(agg.values())

        # define normalizer for matplotlib colors
        normalized = Normalize(vmin=np.min(res), vmax=np.max(res))
        # plot
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        plt.axis('off')
        cmap = cm.get_cmap(cmap)
        if som.topology == "rectangular":
            # axis limits
            ax.set_xlim(-1, som.map_size[1])
            ax.set_ylim(-1, som.map_size[0])
            for pos, hit in zip(som.positions, res):
                # noinspection PyTypeChecker
                rect = RegularPolygon((pos[1], pos[0]), numVertices=4, radius=np.sqrt(0.5),
                                      orientation=np.radians(45), edgecolor='k', facecolor=cmap(normalized(hit)),
                                      alpha=0.2)
                ax.add_patch(rect)

        else:
            # hexagonal topology
            # Horizontal cartesian coords
            hcoord = [c[0] for c in som.positions]
            # Vertical cartersian coords
            vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in som.positions]
            # axis limits
            ax.set_xlim(np.min(hcoord) - 2, np.max(hcoord) + 2)
            ax.set_ylim(np.min(vcoord) - 2, np.max(vcoord) + 2)
            for x, y, hit in zip(hcoord, vcoord, res):
                # noinspection PyTypeChecker
                rect = RegularPolygon((x, y), numVertices=6, radius=2. / 3., orientation=np.radians(30),
                                      edgecolor='k', facecolor=cmap(normalized(hit)), alpha=0.2)
                ax.add_patch(rect)

        # add colorbar
        plt.colorbar(cm.ScalarMappable(norm=normalized, cmap=cmap), ax=ax)
        # show
        plt.show()


def topographic_error(som: BaseSOM, cmap: str = "Reds"):
    """
    Show the topographic error visualization for the map.

    The error count is raised only for the BMU every time the second BMU is not adjacent.

    Parameters:
    -----------
    som: BaseSOM
        The trained SOM where the topographic error should be visualized.
    cmap: str, default = "Reds"
        The string identifier for the matplotlib color map. See
        https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html for more information. The colors
        are scaled linearly.

    Returns:
    --------
    None
    """
    # define function for each SOM type
    types = {
        StandardSOM: __standard_som_topographic_error
    }
    # execute appropriate function
    types[type(som)](som, cmap)
