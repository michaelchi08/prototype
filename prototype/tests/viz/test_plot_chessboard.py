import unittest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from prototype.viz.common import axis_equal_3dplot
from prototype.viz.plot_chessboard import PlotChessboard


class PlotChessboardTest(unittest.TestCase):
    def test_plot_chessboard(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        chessboard = PlotChessboard()
        chessboard.plot(ax)

        # Plot
        # debug = True
        debug = False
        if debug:
            axis_equal_3dplot(ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()
