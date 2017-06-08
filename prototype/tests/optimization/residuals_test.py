import os
import shutil
import unittest

import numpy as np
from numpy import dot

from prototype.utils.math import euler2quat
from prototype.utils.math import euler2rot
from prototype.vision.common import projection_matrix
from prototype.utils.math import nwu2edn
from prototype.vision.dataset import DatasetGenerator
from prototype.optimization.residuals import reprojection_error


class ResidualTest(unittest.TestCase):
    def test_reprojection_error(self):
        debug = False
        dataset = DatasetGenerator()
        dataset.simulate_test_data()

        K = dataset.camera.K
        landmarks = dataset.landmarks
        observed = dataset.observed_landmarks
        robot_states = dataset.robot_states

        for i in range(len(robot_states)):
            x, y, theta = robot_states[i]
            image_point, landmark_id = observed[i][0]
            world_point = nwu2edn(np.array(landmarks[landmark_id]))
            rotation = nwu2edn(np.array([0.0, 0.0, theta]))
            translation = np.array([-y, 0.0, x])

            if debug:
                print("K:", K)
                print("image_point:", image_point)
                print("landmark_id:", landmark_id)
                print("world_point:", world_point)
                print("rotation:", rotation)
                print("translation:", translation)

            residual = reprojection_error(K,
                                          image_point,
                                          rotation,
                                          translation,
                                          world_point)

            self.assertTrue(residual[0] < 0.0001)
            self.assertTrue(residual[1] < 0.0001)
