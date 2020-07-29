"""
This module gathers tests for quantization quality of SOM variants that can be trained in this module.
"""

import unittest
import pandas as pd

from som.maps import StandardSOM
from som.quality.quantization import mqe_m, qe_m, mqe, mmqe


class TestQuantizationStandardSOM(unittest.TestCase):
    def test_rectangular(self):
        data = pd.read_csv('../../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        som.train(data)
        _qe_m = qe_m(som)
        _mqe_m = mqe_m(som)
        _mqe = mqe(som)
        _mmqe = mmqe(som)

    def test_hexagonal(self):
        data = pd.read_csv('../../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5, "hexagonal")
        som.train(data)
        _qe_m = qe_m(som)
        _mqe_m = mqe_m(som)
        _mqe = mqe(som)
        _mmqe = mmqe(som)


if __name__ == '__main__':
    unittest.main()