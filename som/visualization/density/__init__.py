"""
The :mod:`som.visualization.density` module includes visualizations of the distribution of data items on the map.

Each input vector is mapped onto its best-matching unit (BMU).

Let :math:`M` be the set of units in the SOM and :math:`I` be the set of inputs. :math:`W_m` is then defined as

:math:`W_m = \\{ x \\in I \\vert m = \\underset{m' \\in M}{\\mathrm{argmin}}(\\Vert x - m' \\Vert) \}`

the set of inputs where :math:`m` is the BMU.
"""
from ._density import hit_histogram

__all__ = ["hit_histogram"]
