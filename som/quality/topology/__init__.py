"""
The :mod:`som.quality.topology` module includes topology quality measures for trained SOMs.

Let :math:`M` be the set of units in the SOM and :math:`I` be the set of inputs. :math:`W_m` is then defined as

:math:`W_m = \\{ x \\in I \\vert m = \\underset{m' \\in M}{\\mathrm{argmin}}(\\Vert x - m' \\Vert) \}`

the set of inputs where :math:`m` is the BMU.
"""

from ._topology import topographic_error

__all__ = ["topographic_error"]
