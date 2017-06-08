import unittest

import numpy as np

from prototype.control.quadrotor.position import PositionController
from prototype.control.quadrotor.attitude import AttitudeController
from prototype.models.quadrotor import QuadrotorModel


class PositionControllerTest(unittest.TestCase):

    def setUp(self):
        self.quadrotor = QuadrotorModel()
        self.position_controller = PositionController()
        self.attitude_controller = AttitudeController()

    def test_update(self):
        pos_setpoints = np.array([1.0, 2.0, 3.0])

        dt = 0.01
        for i in range(5000):
            # position controller
            pos_actual = np.array([
                self.quadrotor.states[6], self.quadrotor.states[7],
                self.quadrotor.states[8], self.quadrotor.states[2]
            ])
            att_setpoints = self.position_controller.update(
                pos_setpoints, pos_actual, self.quadrotor.states[2], dt)

            # attitude controller
            att_actual = np.array([
                self.quadrotor.states[0], self.quadrotor.states[1],
                self.quadrotor.states[2], self.quadrotor.states[8]
            ])
            motor_inputs = self.attitude_controller.update(
                att_setpoints, att_actual, dt)
            self.quadrotor.update(motor_inputs, dt)

        self.assertTrue(self.quadrotor.states[6] < 1.1)
        self.assertTrue(self.quadrotor.states[6] > 0.9)
        self.assertTrue(self.quadrotor.states[7] < 2.1)
        self.assertTrue(self.quadrotor.states[7] > 1.9)
        self.assertTrue(self.quadrotor.states[8] < 3.1)
        self.assertTrue(self.quadrotor.states[8] > 2.9)
