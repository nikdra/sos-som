import unittest

from som.maps import RectangularSOM


class TestRectangularSOM(unittest.TestCase):

    def test_init(self):
        RectangularSOM((1, 1), 1)
        return True

    def test_train(self):
        som = RectangularSOM((1, 1), 1)
        trained_som = som.train(None)
        self.assertTrue(trained_som.trained)
        # self.assertIsNotNone(trained_som.codebook)


if __name__ == '__main__':
    unittest.main()