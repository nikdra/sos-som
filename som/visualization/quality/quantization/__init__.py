"""
The :mod:`som.visualization.quality.quantization` module includes visualizations of quantization quality measures for
trained SOMs.

We calculate the distance between individual data vectors and the best-matching unit (BMU).

Let :math:`M` be the set of units in the SOM and :math:`I` be the set of inputs. :math:`W_m` is then defined as

:math:`W_m = \\{ x \\in I \\vert m = \\underset{m' \\in M}{\\mathrm{argmin}}(\\Vert x - m' \\Vert) \}`

the set of inputs where :math:`m` is the BMU.
"""
from ._quantization import qe_map

__all__ = ["qe_map"]