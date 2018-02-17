import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.control.quadrotor.position import PositionController
from prototype.control.quadrotor.attitude import AttitudeController
from prototype.models.quadrotor import QuadrotorModel


class QuadrotorTest(unittest.TestCase):
    def setUp(self):
        self.quadrotor = QuadrotorModel()
        self.position_controller = PositionController()
        self.attitude_controller = AttitudeController()

    def test_update(self):
        pos_setpoints = np.array([1.0, 2.0, 3.0])

        time = []
        pos_true = []
        # vel_true = []
        # acc_true = []

        t = 0.0
        dt = 0.01
        for i in range(5000):
            motor_inputs = self.quadrotor.update_pos_controller(
                pos_setpoints,
                dt
            )
            self.quadrotor.update(motor_inputs, dt)

            t += dt
            time.append(t)
            pos_true.append(self.quadrotor.position)

        time = np.array(time)
        pos_true = np.array(pos_true)

        self.assertTrue(self.quadrotor.states[6] < 1.1)
        self.assertTrue(self.quadrotor.states[6] > 0.9)
        self.assertTrue(self.quadrotor.states[7] < 2.1)
        self.assertTrue(self.quadrotor.states[7] > 1.9)
        self.assertTrue(self.quadrotor.states[8] < 3.1)
        self.assertTrue(self.quadrotor.states[8] > 2.9)

        plt.subplot(211)
        plt.plot(pos_true[:, 0], pos_true[:, 1])

        plt.subplot(212)
        plt.plot(time, pos_true[:, 2])

        plt.show()
