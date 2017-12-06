import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.viz.plot_error import plot_error_ellipse


class PlotErrorTests(unittest.TestCase):
    def test_plot_error_ellipse(self):
        # Generate random data
        x = np.random.normal(0, 1, 300)
        s = np.array([2.0, 2.0])
        y1 = np.random.normal(s[0] * x)
        y2 = np.random.normal(s[1] * x)
        data = np.array([y1, y2])

        # Calculate covariance and plot error ellipse
        cov = np.cov(data)
        plot_error_ellipse([0.0, 0.0], cov)

        debug = False
        if debug:
            plt.scatter(data[0, :], data[1, :])
            plt.xlim([-8, 8])
            plt.ylim([-8, 8])
            plt.show()
        plt.clf()
