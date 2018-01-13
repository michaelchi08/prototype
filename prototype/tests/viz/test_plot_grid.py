import unittest

import numpy as np
import matplotlib.pylab as plt
from prototype.viz.plot_grid import plot_grid


class PlotErrorTests(unittest.TestCase):
    def test_plot_error_ellipse(self):
        ax = plt.subplot(111)
        plot_grid(ax, 10, 10)

        # Plot
        debug = True
        # debug = False
        if debug:
            plt.show()
        plt.clf()
