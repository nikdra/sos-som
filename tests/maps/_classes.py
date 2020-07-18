"""
This module gathers tests for SOM variants that can be trained in this module.
"""
import unittest
import pandas as pd

from som.maps import RectangularSOM


class TestRectangularSOM(unittest.TestCase):

    def test_init(self):
        RectangularSOM((1, 1), 1)

    def test_train(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = RectangularSOM((50, 50), 5)
        som.train(data)
        self.assertTrue(som.trained)
        self.assertIsNotNone(som.get_codebook())

    def test_neighborhood_radius_less_than_zero_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM((1, 1), -1)

    def test_neighborhood_radius_equal_zero_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM((1, 1), 0)

    def test_map_size_not_tuple_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM([1, 1], 1)

    def test_map_size_not_len_2_tuple_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM((1, 1, 1), 1)

    def test_map_size_not_int_tuple_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM((1.0, 1.0), 1)

    def test_neighborhood_type_not_supported_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM((1, 1), 1, neighborhood_type="test")

    def test_distance_measure_not_supported_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            RectangularSOM((1, 1), 1, distance_measure="test")

    def test_train_learning_rate_equal_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = RectangularSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, alpha=0)
        self.assertFalse(som.trained)
        self.assertIsNone(som.get_codebook())

    def test_train_learning_rate_less_than_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = RectangularSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, alpha=-0.01)
        self.assertFalse(som.trained)
        self.assertIsNone(som.get_codebook())

    def test_train_data_none_should_raise_value_error(self):
        som = RectangularSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(None)

    def test_train_iterations_equal_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = RectangularSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, iterations=0)

    def test_train_iterations_less_than_zero_should_raise_value_error(self):
        data = pd.read_csv('../data/test_data.csv').drop(['Class'], axis=1)
        som = RectangularSOM((50, 50), 5)
        with self.assertRaises(ValueError):
            som.train(data, iterations=-1)


if __name__ == '__main__':
    unittest.main()

