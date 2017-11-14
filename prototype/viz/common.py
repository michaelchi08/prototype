from math import pi
from math import cos
from math import sin
from math import atan2
from math import sqrt

import numpy as np
from numpy.linalg import eig
import matplotlib.pylab as plt


def plot_error_ellipse(mean, cov):
    """

    Parameters
    ----------
    mean :

    cov :


    Returns
    -------

    """
    # Get eigenvalues and eigenvectors
    [eigenval, eigenvec] = eig(cov)

    # Get largest and smallest eigenvalue
    max_evl = eigenval[1] if eigenval[1] > eigenval[0] else eigenval[0]
    min_evl = eigenval[0] if eigenval[1] > eigenval[0] else eigenval[1]

    # Get largest eigenvalue index which corresponds to largest eigenvector
    # column
    max_evl_col = 1 if eigenval[1] > eigenval[0] else 0

    # Get largest eigenvector
    max_evc = eigenvec[:, max_evl_col]

    # Calculate angle between the x-axis and the largest eigenvector
    angle = atan2(max_evc[1], max_evc[0])

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if angle < 0:
        angle = angle + 2 * pi

    # Plot 99%, 95%, 90% confidence interval error ellipse
    # using the 2 dof Chi-square probabilities
    confidence_intervals = [9.210, 5.991, 4.605]
    for p in confidence_intervals:
        chisquare_val = sqrt(p)
        theta_grid = np.linspace(0, 2 * pi)
        phi = angle

        # Ellipse parameter a and b
        a = chisquare_val * sqrt(max_evl)
        b = chisquare_val * sqrt(min_evl)

        # The ellipse in x and y coordinates
        ellipse_x_r = a * np.cos(theta_grid)
        ellipse_y_r = b * np.sin(theta_grid)

        # Define a rotation matrix
        R = np.matrix([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])

        # Rotate the ellipse to some angle phi
        r_ellipse = R * np.array([[ellipse_x_r], [ellipse_y_r]])

        # Plot error ellipse
        x = np.ravel(r_ellipse[0, :])
        y = np.ravel(r_ellipse[1, :])
        X0, Y0 = mean

        plt.plot(r_ellipse[0, :] + X0, r_ellipse[1, :] + Y0)
        plt.plot(x + X0, y + Y0)
