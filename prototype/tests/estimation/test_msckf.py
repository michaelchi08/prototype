import unittest

import numpy as np

from prototype.estimation.msckf import skew
from prototype.estimation.msckf import C
from prototype.estimation.msckf import CameraState
from prototype.estimation.msckf import MSCKF


class CameraStateTest(unittest.TestCase):
    def test_init(self):
        p_G_C = np.array([1.0, 2.0, 3.0])
        q_GC = np.array([1.0, 2.0, 3.0])
        cam = CameraState(p_G_C, q_GC)

        self.assertTrue(np.array_equal(p_G_C, cam.p_G_C))
        self.assertTrue(np.array_equal(q_GC, cam.q_GC))


class MSCKFTest(unittest.TestCase):
    def setUp(self):
        self.msckf = MSCKF(n_g=0.001 * np.ones(3).reshape((3, 1)),
                           n_a=0.001 * np.ones(3).reshape((3, 1)),
                           n_wg=0.001 * np.ones(3).reshape((3, 1)),
                           n_wa=0.001 * np.ones(3).reshape((3, 1)))
        pass

    def test_skew(self):
        X = skew(np.array([1.0, 2.0, 3.0]))
        X_expected = np.array([[0.0, -3.0, 2.0],
                               [3.0, 0.0, -1.0],
                               [-2.0, 1.0, 0.0]])

        self.assertTrue(np.array_equal(X, X_expected))

    def test_F(self):
        w_hat = np.array([0.0, 0.0, 0.0])
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])
        a_hat = np.array([1.0, 2.0, 3.0])
        w_G = np.array([0.0, 0.0, 0.0])

        F = self.msckf.F(w_hat, q_hat, a_hat, w_G)

        # -- First row --
        self.assertTrue(np.array_equal(F[0:3, 0:3], -skew(w_hat)))
        self.assertTrue(np.array_equal(F[0:3, 3:6], np.ones((3, 3))))

        # -- Third Row --
        self.assertTrue(np.array_equal(F[6:9, 0:3], -C(q_hat).T * skew(a_hat)))
        self.assertTrue(np.array_equal(F[6:9, 6:9], -2.0 * skew(w_G)))
        self.assertTrue(np.array_equal(F[6:9, 9:12], -C(q_hat).T))
        self.assertTrue(np.array_equal(F[6:9, 12:15], -skew(w_G)**2))

        # -- Fifth Row --
        self.assertTrue(np.array_equal(F[12:15, 6:9], np.ones((3, 3))))

    def test_G(self):
        q_hat = np.array([1.0, 0.0, 0.0, 0.0])

        G = self.msckf.G(q_hat)

        # -- First row --
        self.assertTrue(np.array_equal(G[0:3, 0:3], np.ones((3, 3))))

        # -- Second row --
        self.assertTrue(np.array_equal(G[3:6, 3:6], np.ones((3, 3))))

        # -- Third row --
        self.assertTrue(np.array_equal(G[6:9, 6:9], -C(q_hat).T))

        # -- Fourth row --
        self.assertTrue(np.array_equal(G[9:12, 9:12], np.ones((3, 3))))

    def test_prediction_update(self):
        # q_I_W = [0.0, 0.0, 0.0, 0.0]
        # w_W = np.array([[0.0], [0.0], [0.0]])
        # dt = 0.0
        # self.msckf.prediction_update(dt)
        pass

    def test_estimate_features(self):
        # Setup camera states
        N = 10
        cam_states = []
        p_G_C = np.array([1.0, 2.0, 3.0])
        q_GC = np.array([1.0, 2.0, 3.0])
        for i in range(N):
            cam_states.append(CameraState(p_G_C, q_GC))

        # Setup observations
        observations = []

        # Setup imu noise parameters
        noise_params = []

        # Estimate features
        self.msckf.estimate_features(cam_states, observations, noise_params)

    def test_measurement_update(self):
        pass
