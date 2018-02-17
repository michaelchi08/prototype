import unittest

import numpy as np

from prototype.models.two_wheel import two_wheel_2d_model
from prototype.models.two_wheel import two_wheel_2d_deriv
from prototype.models.two_wheel import two_wheel_3d_model


class TwoWheelTest(unittest.TestCase):
    def test_two_wheel_2d_model(self):
        # setup
        t = 0.0
        dt = 0.1
        x = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        u = np.array([1.0, 0.1]).reshape(2, 1)

        # simulate two wheel robot
        data = []
        for i in range(630):
            x = two_wheel_2d_model(x, u, dt)
            t += dt
            data.append(x)

        # convert list of vectors to a 3 x 100 matrix
        data = np.array(data)
        self.assertTrue(data[-1, 0] < 1.0)
        self.assertTrue(data[-1, 0] > 0.0)
        self.assertTrue(data[-1, 1] < 1.0)
        self.assertTrue(data[-1, 1] > 0.0)

        # plot
        # import matplotlib.pylab as plt
        # plt.plot(data[:, 0], data[:, 1])
        # plt.show()

    def test_two_wheel_3d_model(self):
        # Setup
        t = 0.0
        dt = 0.1
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0, 0.1])

        # Simulate two wheel robot
        data = []
        for i in range(630):
            x = two_wheel_3d_model(x, u, dt)
            t += dt
            data.append(x)

        # convert list of vectors to a 3 x 100 matrix
        data = np.array(data)
        self.assertTrue(data[-1, 0] < 1.0)
        self.assertTrue(data[-1, 0] > 0.0)
        self.assertTrue(data[-1, 1] < 1.0)
        self.assertTrue(data[-1, 1] > 0.0)

        # plot
        # import matplotlib.pylab as plt
        # plt.plot(data[:, 0], data[:, 1])
        # plt.show()

    def test_two_wheel_2d_deriv(self):
        two_wheel_2d_deriv()
