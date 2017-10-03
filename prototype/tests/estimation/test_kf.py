import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.estimation.kf import KF


def gaussian_noise(sigma):
    if type(sigma) in [np.matrix, np.array]:
        return sigma**2 * np.random.randn(sigma.shape[0], 1)
    else:
        return sigma**2 * np.random.randn()


def plot_trajectory(state_true, state_estimated):
    plt.plot(state_true[:, 0], state_true[:, 2], color="red")
    plt.scatter(state_estimated[:, 0].tolist()[::10],
                state_estimated[:, 2].tolist()[::10],
                marker="o",
                color="blue")


class KFTest(unittest.TestCase):
    def test_kf(self):
        # Setup
        dt = 0.1
        mu = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
        S = np.eye(4)
        R = np.array([[0.01**2, 0.0, 0.0, 0.0],
                      [0.0, 0.01**2, 0.0, 0.0],
                      [0.0, 0.0, 0.01**2, 0.0],
                      [0.0, 0.0, 0.0, 0.01**2]])
        Q = np.array([[0.4**2, -0.1],
                      [-0.1, 0.1**2]])
        A = np.array([[1.0, 0.0975, 0.0, 0.0],
                      [0.0, 0.9512, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0975],
                      [0.0, 0.0, 0.0, 0.9512]])
        B = np.array([[0.0025, 0.0],
                      [0.0488, 0.0],
                      [0.0, 0.0025],
                      [0.0, 0.0488]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        kf = KF(mu=mu, S=S, R=R, Q=Q)

        # Simulation parameters
        t_end = 100
        T = np.arange(0, t_end, dt)
        x = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)

        # Simulation
        state_true = []
        state_estimated = []

        for t in T:
            # Update state
            u = np.array([[1.0], [1.0]])
            x = np.dot(A, x) + np.dot(B, u)
            state_true.append(x)

            # Take measurement + noise
            d = gaussian_noise(kf.Q)
            y = np.dot(C, x) + d

            # KF
            kf.prediction_update(A, B, u, dt)
            kf.measurement_update(C, y)

            # Store true and estimated
            state_true.append(x)
            state_estimated.append(kf.mu)

        # Convert from list to numpy array then to matrix
        state_true = np.array(state_true)
        state_true = np.matrix(state_true)
        state_estimated = np.array(state_estimated)
        state_estimated = np.matrix(state_estimated)

        # Plot trajectory
        debug = True
        if debug:
            plot_trajectory(state_true, state_estimated)
            plt.show()
