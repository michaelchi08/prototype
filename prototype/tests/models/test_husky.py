import unittest
from math import pi
from math import ceil

import numpy as np
import matplotlib.pylab as plt

from prototype.models.husky import HuskyModel
from prototype.control.utils import circle_trajectory


class HuskyModelTest(unittest.TestCase):
    def test_init(self):
        model = HuskyModel()
        self.assertTrue(np.array_equal(model.p_G, np.zeros((3, 1))))
        self.assertTrue(np.array_equal(model.v_G, np.zeros((3, 1))))
        self.assertTrue(np.array_equal(model.a_G, np.zeros((3, 1))))
        self.assertTrue(np.array_equal(model.rpy_G, np.zeros((3, 1))))
        self.assertTrue(np.array_equal(model.w_B, np.zeros((3, 1))))

    def test_update(self):
        # Setup
        model = HuskyModel()
        t = 0.1
        dt = 0.1

        # Calculate desired inputs for a circle trajectory
        circle_r = 10.0
        circle_vel = 1.0
        circle_w = circle_trajectory(circle_r, circle_vel)
        circumference = 2 * pi * circle_r
        max_iter = ceil((circumference / circle_vel) / dt)

        # Inputs
        v_B = np.array([[circle_vel], [0.0], [0.0]])
        w_B = np.array([[0.0], [0.0], [circle_w]])

        # Simulate
        time = np.array([0.0])
        pos_true = np.zeros((3, 1))
        vel_true = np.zeros((3, 1))
        acc_true = np.zeros((3, 1))
        rpy_true = np.zeros((3, 1))
        imu_acc_true = np.zeros((3, 1))
        imu_gyr_true = np.zeros((3, 1))

        for i in range(max_iter):
            model.update(v_B, w_B, dt)

            pos_true = np.hstack((pos_true, model.p_G))
            vel_true = np.hstack((vel_true, model.v_G))
            acc_true = np.hstack((acc_true, model.a_G))
            rpy_true = np.hstack((rpy_true, model.rpy_G))
            imu_acc_true = np.hstack((imu_acc_true, model.a_B))
            imu_gyr_true = np.hstack((imu_gyr_true, model.w_B))
            time = np.hstack((time, t))
            t += dt

        self.assertTrue(abs(pos_true[0, -1]) < 0.1)
        self.assertTrue(abs(pos_true[1, -1]) < 0.1)
        self.assertTrue(abs(pos_true[2, -1]) < 0.1)

        # Plot
        # debug = False
        debug = True
        if debug:
            # plt.figure()
            # plt.plot(pos_true[0, :], pos_true[1, :])
            # plt.axis("equal")
            # plt.legend(loc=0)
            # plt.title("Position")

            # plt.figure()
            # plt.plot(time, vel_true[0, :], label="vx")
            # plt.plot(time, vel_true[1, :], label="vy")
            # plt.plot(time, vel_true[2, :], label="vz")
            # plt.legend(loc=0)
            # plt.title("Velocity")

            # plt.figure()
            # plt.plot(time, acc_true[0, :], label="ax")
            # plt.plot(time, acc_true[1, :], label="ay")
            # plt.plot(time, acc_true[2, :], label="az")
            # plt.legend(loc=0)
            # plt.title("Acceleration")

            plt.figure()
            plt.plot(time, imu_acc_true[0, :], label="ax")
            plt.plot(time, imu_acc_true[1, :], label="ay")
            plt.plot(time, imu_acc_true[2, :], label="az")
            plt.legend(loc=0)
            plt.title("Accelerometer")

            plt.show()
