from math import pi

import cv2
import numpy as np

from prototype.utils.euler import euler2rot


class AprilGrid:
    """ AprilGrid

    Attributes
    ----------
    nb_rows : float
        Number of rows
    nb_cols : float
        Number of columns
    R_G : np.array
        Rotation in global frame
    t_G : np.array
        Translation in global frame
    grid_points : np.array
        Matrix of Nx2 grid points

    Parameters
    ----------
    nb_rows : float
        Number of rows
    nb_cols : float
        Number of columns
    R_G : np.array
        Rotation in global frame
    t_G : np.array
        Translation in global frame

    """
    def __init__(self, **kwargs):
        self.nb_rows = kwargs.get("nb_rows", 10)
        self.nb_cols = kwargs.get("nb_cols", 10)
        self.square_size = kwargs.get("square_size", 0.1)

        center_x = ((self.nb_cols - 1) * self.square_size) / 2.0
        center_y = ((self.nb_rows - 1) * self.square_size) / 2.0
        self.center = np.array([center_x, center_y])

        self.R_BG = kwargs.get("R_BG", euler2rot([-pi / 2, 0.0, -pi / 2], 321))
        self.t_G = kwargs.get("t_G", np.zeros(3))

        # Grid_points as a list of (x, y) points
        # starting from top left corner to bottom right
        self.grid_points = []
        for i in range(self.nb_rows):
            for j in range(self.nb_rows):
                self.grid_points.append([i, j])
        self.grid_points = self.square_size * np.array(self.grid_points)
