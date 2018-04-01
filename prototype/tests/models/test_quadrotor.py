import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.control.quadrotor.position import PositionController
from prototype.control.quadrotor.attitude import AttitudeController
from prototype.models.quadrotor import QuadrotorModel


class QuadrotorTest(unittest.TestCase):
    # def test_update_position_control(self):
    #     pos_setpoints = np.array([1.0, 2.0, 3.0])
    #
    #     time = []
    #     pos_true = []
    #     # vel_true = []
    #     # acc_true = []
    #     quadrotor = QuadrotorModel()
    #
    #     t = 0.0
    #     dt = 0.01
    #     for i in range(5000):
    #         motor_inputs = quadrotor.update_position_controller(
    #             pos_setpoints,
    #             dt
    #         )
    #         quadrotor.update(motor_inputs, dt)
    #
    #         t += dt
    #         time.append(t)
    #         pos_true.append(quadrotor.position)
    #
    #     time = np.array(time)
    #     pos_true = np.array(pos_true)
    #
    #     self.assertTrue(quadrotor.states[6] < 1.1)
    #     self.assertTrue(quadrotor.states[6] > 0.9)
    #     self.assertTrue(quadrotor.states[7] < 2.1)
    #     self.assertTrue(quadrotor.states[7] > 1.9)
    #     self.assertTrue(quadrotor.states[8] < 3.1)
    #     self.assertTrue(quadrotor.states[8] > 2.9)
    #
    #     plt.subplot(211)
    #     plt.plot(pos_true[:, 0], pos_true[:, 1])
    #
    #     plt.subplot(212)
    #     plt.plot(time, pos_true[:, 2])
    #
    #     debug = False
    #     # debug = True
    #     if debug:
    #         plt.show()

    def test_update_waypoint_control(self):
        waypoints = np.array([[0.0, 0.0, 5.0],
                              [5.0, 0.0, 5.0],
                              [5.0, 5.0, 5.0],
                              [0.0, 5.0, 5.0],
                              [0.0, 0.0, 5.0]])
        # waypoints = np.array([[0.0, 0.0, 5.0],
        #                       [1.0, 0.0, 5.0],
        #                       [2.0, 0.0, 5.0]])
        look_ahead_dist = 0.1

        time = []
        pos_true = []
        # vel_true = []
        # acc_true = []
        quadrotor = QuadrotorModel(ctrl_mode="WAYPOINT_MODE",
                                   waypoints=waypoints,
                                   look_ahead_dist=look_ahead_dist)
        quadrotor.states[6:9] = np.array([0.0, 0.0, 5.0])

        t = 0.0
        dt = 0.001
        for i in range(40000):
            motor_inputs = quadrotor.update_waypoint_controller(dt)
            quadrotor.update(motor_inputs, dt)

            t += dt
            time.append(t)
            pos_true.append(quadrotor.position)

        time = np.array(time)
        pos_true = np.array(pos_true)

        plt.subplot(211)
        plt.plot(pos_true[:, 0], pos_true[:, 1])
        plt.xlim([-0.5, 6.0])
        plt.ylim([-0.5, 6.0])

        plt.subplot(212)
        plt.plot(time, pos_true[:, 2])
        plt.ylim([4.0, 6.0])

        # debug = False
        debug = True
        if debug:
            plt.show()
