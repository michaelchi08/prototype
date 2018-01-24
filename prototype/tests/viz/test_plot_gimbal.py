import unittest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from prototype.utils.utils import deg2rad
from prototype.utils.euler import euler2rot
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.viz.common import axis_equal_3dplot
from prototype.models.gimbal import dh_transform


class PlotGimbalTest(unittest.TestCase):
    def test_plot_elbow_manipulator(self):
        # Link angles
        link1_theta = 20.0
        link2_theta = 10.0

        # Create base frame
        rpy = [0.0, 0.0, 0.0]
        R_BG = euler2rot([deg2rad(i) for i in rpy], 321)
        t_G_B = np.array([2.0, 1.0, 0.0])
        T_GB = np.array([[R_BG[0, 0], R_BG[0, 1], R_BG[0, 2], t_G_B[0]],
                         [R_BG[1, 0], R_BG[1, 1], R_BG[1, 2], t_G_B[1]],
                         [R_BG[2, 0], R_BG[2, 1], R_BG[2, 2], t_G_B[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        # Create DH Transforms
        T_B1 = dh_transform(deg2rad(link1_theta), 0.0, 1.0, 0.0)
        T_12 = dh_transform(deg2rad(link2_theta), 0.0, 1.0, 0.0)

        # Create Transforms
        T_G1 = np.dot(T_GB, T_B1)
        T_G2 = np.dot(T_GB, np.dot(T_B1, T_12))

        # Transform from origin to end-effector
        links = []
        links.append(T_G1)
        links.append(T_G2)

        # Plot first link
        # debug = True
        debug = False
        if debug:
            plt.figure()
            plt.plot([T_GB[0, 3], links[0][0, 3]],
                     [T_GB[1, 3], links[0][1, 3]])

            plt.plot([links[0][0, 3], links[1][0, 3]],
                     [links[0][1, 3], links[1][1, 3]])

            obs = np.array([1.0, 1.0, 0.0, 1.0])
            obs_end = np.dot(T_G2, obs)
            plt.plot(obs_end[0], obs_end[1], marker="x")

            plt.xlim([0.0, 6.0])
            plt.ylim([0.0, 6.0])
            plt.show()

    def test_plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        gimbal = PlotGimbal()
        gimbal.set_attitude([10.0, 20.0, 0.0])
        gimbal.plot(ax)

        # Plot
        debug = True
        # debug = False
        if debug:
            axis_equal_3dplot(ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()
