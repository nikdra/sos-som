import numpy as np

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt

from som.maps import StandardSOM, BaseSOM
from som.quality.quantization import qe_m, mqe_m


def __standard_som_qe_m(som: StandardSOM, cmap: str):
    """
    Plot the quantization error for each unit for a standard SOM.
    Plot depends on the topology of the neighborhood (hexagonal or rectangular).

    Parameters:
    -----------
    som: StandardSOM
        The SOM for which the quantization error should be plotted.
    cmap: str
        The matplotlib color map for the map.

    Returns:
    --------
    None
    """
    if som.codebook is not None and som.trained:
        # get qe for each unit
        qe_ms = qe_m(som)
        # define normalizer for matplotlib colors
        normalized = Normalize(vmin=np.min(qe_ms), vmax=np.max(qe_ms))
        # plot
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        plt.axis('off')
        cmap = cm.get_cmap(cmap)
        if som.topology == "rectangular":
            # axis limits
            ax.set_xlim(-1, som.map_size[1])
            ax.set_ylim(-1, som.map_size[0])
            for pos, hit in zip(som.positions, qe_ms):
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
            for x, y, hit in zip(hcoord, vcoord, qe_ms):
                # noinspection PyTypeChecker
                rect = RegularPolygon((x, y), numVertices=6, radius=2. / 3., orientation=np.radians(30),
                                      edgecolor='k', facecolor=cmap(normalized(hit)), alpha=0.2)
                ax.add_patch(rect)

        # add colorbar
        plt.colorbar(cm.ScalarMappable(norm=normalized, cmap=cmap), ax=ax)
        # show
        plt.show()


def __standard_som_mqe_m(som: StandardSOM, cmap: str):
    """
    Plot the mean quantization error for each unit for a standard SOM.
    Plot depends on the topology of the neighborhood (hexagonal or rectangular).

    Parameters:
    -----------
    som: StandardSOM
        The SOM for which the mean quantization error should be plotted.
    cmap: str
        The matplotlib color map for the map.

    Returns:
    --------
    None
    """
    if som.codebook is not None and som.trained:
        # get qe for each unit
        mqe_ms = mqe_m(som)
        # define normalizer for matplotlib colors
        normalized = Normalize(vmin=np.min(mqe_ms), vmax=np.max(mqe_ms))
        # plot
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        plt.axis('off')
        cmap = cm.get_cmap(cmap)
        if som.topology == "rectangular":
            # axis limits
            ax.set_xlim(-1, som.map_size[1])
            ax.set_ylim(-1, som.map_size[0])
            for pos, hit in zip(som.positions, mqe_ms):
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
            for x, y, hit in zip(hcoord, vcoord, mqe_ms):
                # noinspection PyTypeChecker
                rect = RegularPolygon((x, y), numVertices=6, radius=2. / 3., orientation=np.radians(30),
                                      edgecolor='k', facecolor=cmap(normalized(hit)), alpha=0.2)
                ax.add_patch(rect)

        # add colorbar
        plt.colorbar(cm.ScalarMappable(norm=normalized, cmap=cmap), ax=ax)
        # show
        plt.show()


def qe_map(som: BaseSOM, cmap: str = "Reds"):
    """
    Show the quantization error for each unit in the map.

    Parameters:
    -----------
    som: BaseSOM
        The trained SOM where the quantization error should be visualized.
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
        StandardSOM: __standard_som_qe_m
    }
    # execute appropriate function
    types[type(som)](som, cmap)


def mqe_map(som: BaseSOM, cmap: str = "Reds"):
    """
    Show the mean quantization error for each unit in the map.

    Parameters:
    -----------
    som: BaseSOM
        The trained SOM where the mean quantization error should be visualized.
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
        StandardSOM: __standard_som_mqe_m
    }
    # execute appropriate function
    types[type(som)](som, cmap)
