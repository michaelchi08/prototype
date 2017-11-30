import unittest

import numpy as np
from numpy import allclose
from numpy import array_equal

from prototype.estimation.msckf.camera_state import CameraState


class CameraStateTest(unittest.TestCase):
    def test_init(self):
        p_G = np.array([1.0, 2.0, 3.0])
        q_CG = np.array([1.0, 0.0, 0.0, 0.0])
        cam = CameraState(0, q_CG, p_G)

        self.assertEqual(cam.frame_id, 0)
        self.assertTrue(array_equal(p_G, cam.p_G.ravel()))
        self.assertTrue(array_equal(q_CG, cam.q_CG.ravel()))

    def test_correct(self):
        p_G = np.array([1.0, 2.0, 3.0])
        q_CG = np.array([0.0, 0.0, 0.0, 1.0])
        cam = CameraState(0, q_CG, p_G)

        dtheta_CG = np.array([[0.1], [0.2], [0.3]])
        dp_G = np.array([[0.1], [0.2], [0.3]])
        dx = np.block([[dtheta_CG], [dp_G]])
        cam.correct(dx)

        # print(cam)
        expected_p_G = np.array([1.1, 2.2, 3.3])
        expected_q_CG = np.array([0.05, 0.1, 0.15, 0.98])

        self.assertEqual(cam.frame_id, 0)
        self.assertTrue(allclose(expected_p_G, cam.p_G.ravel()))
        self.assertTrue(allclose(expected_q_CG, cam.q_CG.ravel(), rtol=1e-2))

    def test_str(self):
        p_G = np.array([1.0, 2.0, 3.0])
        q_CG = np.array([0.0, 0.0, 0.0, 1.0])
        cam = CameraState(0, q_CG, p_G)

        cam_str = str(cam)
        expected = """
Camera state:
frame_id: 0
q: [ 0.  0.  0.  1.]
p: [ 1.  2.  3.]
        """
        self.assertEqual(cam_str.strip(), expected.strip())
