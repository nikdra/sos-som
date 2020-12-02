import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt

from som.maps import StandardSOM


def __standard_som_hit_histogram(som: StandardSOM, cmap: str):
    """
    Plot the hit histogram for a standard SOM. Plot depends on the topology of the neighborhood (hexagonal or
    rectangular).

    Parameters:
    -----------
    som: StandardSOM
        The SOM for which the hit histogram should be plotted.
    cmap: str
        The matplotlib color map for the map.

    Returns:
    --------
    None
    """
    if som.codebook is not None and som.trained:
        first_bmus = som.get_first_bmus()
        # init hit result array
        hits = np.zeros(len(som.positions))
        # aggregate for each position with hits
        agg = {k: len(v) for k, v in first_bmus.items()}
        # fill up values at the resp. positions
        hits[list(agg.keys())] = list(agg.values())
        # define normalizer for matplotlib colors
        normalized = Normalize(vmin=np.min(hits), vmax=np.max(hits))
        # plot
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        plt.axis('off')
        cmap = cm.get_cmap(cmap)
        if som.topology == "rectangular":
            # axis limits
            ax.set_xlim(-1, som.map_size[1])
            ax.set_ylim(-1, som.map_size[0])
            for pos, hit in zip(som.positions, hits):
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
            for x, y, hit in zip(hcoord, vcoord, hits):
                # noinspection PyTypeChecker
                rect = RegularPolygon((x, y), numVertices=6, radius=2. / 3., orientation=np.radians(30),
                                      edgecolor='k', facecolor=cmap(normalized(hit)), alpha=0.2)
                ax.add_patch(rect)

        # add colorbar
        plt.colorbar(cm.ScalarMappable(norm=normalized, cmap=cmap), ax=ax)
        # show
        plt.show()


def hit_histogram(som, cmap: str = "Reds"):
    """
    Show the hit histogram for the map

    Parameters:
    -----------
    som: BaseSOM
        The trained SOM where the hit histogram should be visualized.
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
        StandardSOM: __standard_som_hit_histogram
    }
    # execute appropriate function
    types[type(som)](som, cmap)


