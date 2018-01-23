import numpy as np

from prototype.calibration.chessboard import Chessboard


class PlotChessboard:
    def __init__(self):
        self.chessboard = Chessboard()

    def plot(self, ax):
        """ Plot chessboard

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axes

        """
        grid_points = self.chessboard.grid_points
        R_BG = self.chessboard.R_BG
        t_G = self.chessboard.t_G
        # R_BG = euler2rot([deg2rad(i) for i in [0.0, 20.0, 0.0]], 321)
        # t_G = np.array([0.0, 0.0, 0.0])

        for point in grid_points:
            point = point - self.chessboard.center
            p = np.array([point[0], point[1], 0.0]) + t_G
            p_G = np.dot(R_BG, p)
            ax.plot([p_G[0]], [p_G[1]], [p_G[2]], marker="o", color="red")
