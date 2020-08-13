"""
This module gathers tests for SOM variants that can be trained in this module.
"""
import unittest
import pandas as pd

from som.maps import StandardSOM


class TestStandardSOM(unittest.TestCase):

    def test_init(self):
        StandardSOM((1, 1), 1)

    def test_train_rectangular(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        som.train(data)
        self.assertTrue(som.trained)
        self.assertIsNotNone(som.codebook)
        self.assertIsNotNone(som.bmu_indices)
        self.assertIsNotNone(som.bmu_distances)
        self.assertIsNotNone(som.get_first_bmus())
        self.assertIsNotNone(som.get_second_bmus())
        som.hit_histogram()

    def test_train_hexagonal(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5, "hexagonal")
        som.train(data)
        self.assertTrue(som.trained)
        self.assertIsNotNone(som.codebook)
        self.assertIsNotNone(som.bmu_indices)
        self.assertIsNotNone(som.bmu_distances)
        self.assertIsNotNone(som.get_first_bmus())
        self.assertIsNotNone(som.get_second_bmus())
        som.hit_histogram()

    def test_neighborhood_radius_less_than_zero_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM((1, 1), -1)

    def test_neighborhood_radius_equal_zero_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM((1, 1), 0)

    def test_map_size_not_tuple_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM([1, 1], 1)

    def test_map_size_not_len_2_tuple_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM((1, 1, 1), 1)

    def test_map_size_not_int_tuple_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM((1.0, 1.0), 1)

    def test_neighborhood_type_not_supported_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM((1, 1), 1, neighborhood_type="test")

    def test_distance_measure_not_supported_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            StandardSOM((1, 1), 1, distance_measure="test")

    def test_train_learning_rate_equal_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, alpha=0)
        self.assertFalse(som.trained)
        self.assertIsNone(som.codebook)

    def test_train_learning_rate_less_than_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, alpha=-0.01)
        self.assertFalse(som.trained)
        self.assertIsNone(som.codebook)

    def test_train_data_none_should_raise_value_error(self):
        som = StandardSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(None)

    def test_train_iterations_equal_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, iterations=0)

    def test_train_iterations_less_than_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = StandardSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, iterations=-1)


if __name__ == '__main__':
    unittest.main()
