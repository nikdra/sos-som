"""
This module gathers tests for quantization quality visualizations of SOM variants that can be trained in this module.
"""
import unittest
import pandas as pd

from som.maps import StandardSOM
from som.visualization.quality.quantization import qe_map


class TestStandardSOM(unittest.TestCase):
    def test_rectangular(self):
        data = pd.read_csv('../../../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        som.train(data)
        self.assertTrue(som.trained)
        self.assertIsNotNone(som.codebook)
        self.assertIsNotNone(som.bmu_indices)
        self.assertIsNotNone(som.bmu_distances)
        self.assertIsNotNone(som.get_first_bmus())
        self.assertIsNotNone(som.get_second_bmus())
        qe_map(som)

    def test_hexagonal(self):
        data = pd.read_csv('../../../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5, "hexagonal")
        som.train(data)
        self.assertTrue(som.trained)
        self.assertIsNotNone(som.codebook)
        self.assertIsNotNone(som.bmu_indices)
        self.assertIsNotNone(som.bmu_distances)
        self.assertIsNotNone(som.get_first_bmus())
        self.assertIsNotNone(som.get_second_bmus())
        qe_map(som)


if __name__ == '__main__':
    unittest.main()