import unittest
from math import pi

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from prototype.utils.utils import deg2rad
from prototype.utils.euler import euler2rot
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.viz.common import axis_equal_3dplot
from prototype.models.gimbal import dh_transform
from prototype.models.gimbal import GimbalModel


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
        # Setup gimbal model
        # -- tau (x, y, z, roll, pitch, yaw)
        tau_s = np.array([-0.045, -0.085, 0.08, 0.0, 0.0, 0.0])
        tau_d = np.array([0.01, 0.0, -0.03, pi / 2.0, 0.0, -pi / 2.0])
        # -- link (theta, alpha, a, d)
        alpha = pi / 2.0
        offset = -pi / 2.0
        roll = 0.0
        pitch = 0.1
        link1 = np.array([offset + roll, alpha, 0.0, 0.045])
        link2 = np.array([pitch, 0.0, 0.0, 0.0])
        # -- Gimbal model
        gimbal_model = GimbalModel(
            tau_s=tau_s,
            tau_d=tau_d,
            link1=link1,
            link2=link2
        )
        # T_ds = gimbal_model.T_ds()
        # p_s = np.array([1.0, 0.0, 0.0, 1.0])

        # Plot gimbal model
        plot = PlotGimbal(gimbal=gimbal_model,
                          show_static_frame=True,
                          show_base_frame=True,
                          show_end_frame=True,
                          show_dynamic_frame=True)
        plot.set_attitude([0.0, 0.0])
        plot.plot()

        # Plot
        debug = True
        # debug = False
        if debug:
            plt.show()
