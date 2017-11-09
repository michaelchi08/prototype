import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.estimation.ekf import EKF
from prototype.utils.utils import deg2rad

from prototype.models.two_wheel import two_wheel_2d_model
from prototype.models.two_wheel import two_wheel_2d_linearized_model
from prototype.models.two_wheel import two_wheel_2d_measurement_model
from prototype.models.two_wheel import two_wheel_2d_measurement_linearized_model


def gaussian_noise(sigma):
    if type(sigma) in [np.matrix, np.array]:
        return sigma**2 * np.random.randn(sigma.shape[0], 1)
    else:
        return sigma**2 * np.random.randn()


def plot_trajectory(state_true, state_estimated):
    plt.plot(state_true[:, 0], state_true[:, 1], color="red")
    plt.scatter(state_estimated[:, 0].tolist()[::50],
                state_estimated[:, 1].tolist()[::50],
                marker="o",
                color="blue")


class EKFTest(unittest.TestCase):
    def test_sandbox(self):
        # Setup
        dt = 0.1
        mu = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        R = np.array([[0.05**2, 0.0, 0.0],
                      [0.0, 0.05**2, 0.0],
                      [0.0, 0.0, deg2rad(0.5)**2]])
        Q = np.array([[0.1**2, 0.0, 0.0],
                      [0.0, 0.1**2, 0.0],
                      [0.0, 0.0, deg2rad(10)**2]])
        S = np.eye(3)
        ekf = EKF(mu=mu, R=R, Q=Q, S=S)

        # Simulation parameters
        t_end = 1000
        T = np.arange(0, t_end, dt)
        x = np.array([[0.0], [0.0], [0.0]]).reshape(3, 1)
        u = np.array([[1.0], [0.1]]).reshape(2, 1)

        # Simulation
        state_true = []
        state_estimated = []

        for t in T:
            # Update state
            x = two_wheel_2d_model(x, u, dt)
            state_true.append(x)

            # Take measurement + noise
            d = gaussian_noise(ekf.Q)
            y = x + d

            # EKF
            g = two_wheel_2d_model(ekf.mu, u, dt)
            G = two_wheel_2d_linearized_model(ekf.mu, u, dt)
            ekf.prediction_update(g, G, dt)

            h = two_wheel_2d_measurement_model(ekf.mu)
            H = two_wheel_2d_measurement_linearized_model(ekf.mu)
            ekf.measurement_update(y, h, H)

            # Store true and estimated
            state_true.append(x)
            state_estimated.append(ekf.mu)

        # Convert from list to numpy array then to matrix
        state_true = np.array(state_true)
        state_true = np.matrix(state_true)
        state_estimated = np.array(state_estimated)
        state_estimated = np.matrix(state_estimated)

        # Plot trajectory
        debug = False
        if debug:
            plot_trajectory(state_true, state_estimated)
            plt.show()
