import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.models.imu import generate_signal
from prototype.models.imu import IMUSim


class IMUTest(unittest.TestCase):
    def test_update(self):
        imu = IMUSim()
        v_W = np.array([0.0, 0.0, 0.0])
        w_W = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        x = []
        for i in range(1000):
            a_W, w_W = imu.update(v_W, w_W, dt)
            x.append(a_W[0])

        # Check standard deviation
        sigma = 0.01
        print(abs(sigma - np.std(x, ddof=1)) < 0.01)

        # Plot imu data
        debug = False
        if debug:
            plt.plot(range(1000), x)
            plt.show()

    # def test_generate_signal(self):
    #     dt = 1e-2
    #     x = generate_signal(1000000, dt, 0.1, 0.05, 0.002)
    #
    #     plt.plot(range(x.shape[0]), x)
    #     plt.show()
