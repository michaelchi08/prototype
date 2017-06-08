import unittest

import numpy as np

from prototype.utils.math import deg2rad
from prototype.control.quadrotor.attitude import AttitudeController
from prototype.models.quadrotor import QuadrotorModel


class AttitudeControllerTest(unittest.TestCase):

    def setUp(self):
        self.quadrotor = QuadrotorModel()
        self.attitude_controller = AttitudeController()

    def test_update(self):
        setpoints = np.array(
            [deg2rad(10.0), deg2rad(10.0), deg2rad(10.0), 0.5])

        dt = 0.001
        for i in range(1000):
            actual = np.array([
                self.quadrotor.states[0], self.quadrotor.states[1],
                self.quadrotor.states[2], self.quadrotor.states[8]
            ])
            motor_inputs = self.attitude_controller.update(
                setpoints, actual, dt)
            self.quadrotor.update(motor_inputs, dt)
