import unittest

import numpy as np

from prototype.vision.common import focal_length
from prototype.vision.common import camera_intrinsics
from prototype.vision.camera.distortion_model import project_pinhole_equi


class DistortionModelTest(unittest.TestCase):
    def test_project_pinhole_equi(self):
        image_width = 640
        image_height = 480
        fov = 120
        fx, fy = focal_length(image_width, image_height, fov)
        cx, cy = (image_width / 2.0, image_height / 2.0)
        K = camera_intrinsics(fx, fy, cx, cy)

        D = np.array([0.0, 0.0, 0.0, 0.0])

        X_c = np.array([0.0001, 0.0001, 1.0])
        result = project_pinhole_equi(X_c, K, D)
        print(result)
