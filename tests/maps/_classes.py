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


if __name__ == '__main__':
    unittest.main()
