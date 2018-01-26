import matplotlib.pyplot as plt

from prototype.viz.common import axis_equal_3dplot
from prototype.calibration.chessboard import Chessboard


class PlotChessboard:
    def __init__(self, **kwargs):
        self.chessboard = kwargs.get("chessboard", Chessboard())

    def plot(self, ax=None):
        """ Plot chessboard

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        # Plot chessboard points
        for p in self.chessboard.grid_points3d:
            ax.plot([p[0]], [p[1]], [p[2]], marker="o", color="red")

        # Plot settings
        axis_equal_3dplot(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        return ax
