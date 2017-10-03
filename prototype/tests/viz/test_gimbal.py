import unittest
from math import cos
from math import sin

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from prototype.viz.gimbal import GimbalPlot
from prototype.viz.gimbal import dh_transform_matrix
from prototype.utils.utils import deg2rad


def axis_equal_3dplot(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


class GimbalPlotTest(unittest.TestCase):
    def test_plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        gimbal = GimbalPlot()

        gimbal.plot(ax)

        debug = False
        if debug:
            axis_equal_3dplot(ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()

    def test_plot_elbow_manipulator(self):
        # Link angles
        link1_theta = 10.0
        link2_theta = 45.0

        # Create DH Transforms
        T_1 = dh_transform_matrix(deg2rad(link1_theta), 0.0, 1.0, 0.0)
        T_2 = dh_transform_matrix(deg2rad(link2_theta), 0.0, 1.0, 0.0)

        # Transform from origin to end-effector
        origin = np.array([0.0, 0.0, 0.0, 1.0]).reshape((4, 1))
        links = []
        links.append(T_1 * origin)
        links.append(T_1 * T_2 * origin)

        # Plot first link
        debug = False
        if debug:
            plt.plot([origin[0], links[0][0, 0]],
                     [origin[1], links[0][1, 0]])

            plt.plot([links[0][0, 0], links[1][0, 0]],
                     [links[0][1, 0], links[1][1, 0]])

            plt.xlim([0.0, 2.0])
            plt.ylim([0.0, 2.0])
            plt.show()
