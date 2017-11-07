import unittest
import numpy as np
import matplotlib.pylab as plt

from prototype.vision.ransac import RANSAC


class RANSACTest(unittest.TestCase):
    def setUp(self):
        self.ransac = RANSAC()

        # Setup test data based on line equation y = mx + c
        self.m_true = 1.0  # Gradient
        self.c_true = 2.0  # Intersection
        x = np.linspace(0.0, 10.0, num=100)
        y = self.m_true * x + self.c_true

        # Add guassian noise to data
        for i in range(len(y)):
            y[i] += np.random.normal(0.0, 0.2)

        # Add outliers to 30% of data
        for i in range(int(len(y) * 0.3)):
            idx = np.random.randint(0, len(y))
            y[idx] += np.random.normal(0.0, 1.0)

        self.data = np.array([x, y])

    def test_sample(self):
        sample = self.ransac.sample(self.data)
        self.assertEqual(2, len(sample))

    def test_distance(self):
        sample = self.ransac.sample(self.data)
        dist = self.ransac.compute_distance(sample, self.data)
        self.assertEqual((1, 100), dist.shape)

    def test_compute_inliers(self):
        sample = self.ransac.sample(self.data)
        dist = self.ransac.compute_distance(sample, self.data)
        self.ransac.compute_inliers(dist)

    def test_optimize(self):
        m_pred, c_pred = self.ransac.optimize(self.data)

        print("m_true: ", self.m_true)
        print("m_pred: ", m_pred)
        print("c_true: ", self.c_true)
        print("c_pred: ", c_pred)

        self.assertTrue(abs(m_pred - self.m_true) < 0.5)
        self.assertTrue(abs(c_pred - self.c_true) < 0.5)

        # Plot RANSAC optimized result
        debug = False
        if debug:
            x = np.linspace(0.0, 10.0, num=100)
            y = m_pred * x + c_pred
            plt.scatter(self.data[0, :], self.data[1, :])
            plt.plot(x, y)
            plt.show()
