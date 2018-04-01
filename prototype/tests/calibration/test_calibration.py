import unittest
from os.path import join
from math import pi

import numpy as np

import prototype.tests as test
from prototype.models.gimbal import GimbalModel
from prototype.calibration.calibration import GimbalCalibrator


class GimbalCalibratorTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.calib = GimbalCalibrator(
            data_path=self.data_path,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv",
            nb_rows=6,
            nb_cols=7,
            square_size=0.0285
        )

    def test_setup_problem(self):
        x, Z, K_s, K_d, D_s, D_d = self.calib.setup_problem()

        # self.assertEqual((28, ), x.shape)
        # self.assertEqual(5, len(Z))

    def test_reprojection_error(self):
        x, Z, K_s, K_d, D_s, D_d = self.calib.setup_problem()
        args = [Z, K_s, K_d, D_s, D_d]

        result = self.calib.reprojection_error(x, *args)
        print(result)

    def test_optimize(self):
        self.calib.optimize()

    def test_optimize_preprocessed(self):
        gimbal_model = GimbalModel(
            tau_s=np.array([0.045, 0.075, -0.085, 0.0, 0.0, pi / 2.0]),
            tau_d=np.array([0.0, 0.015, 0.0, 0.0, 0.0, -pi / 2.0]),
            w1=np.array([0.0, 0.0, 0.075]),
            w2=np.array([0.0, 0.0, 0.0])
        )
        # gimbal_model.set_attitude([deg2rad(0), deg2rad(0)])
        # plot_gimbal = PlotGimbal(gimbal=gimbal_model)
        # plot_gimbal.plot()
        # plt.show()

        data_path = "/home/chutsu/Dropbox/calib_data"
        calib = GimbalCalibrator(
            preprocessed=True,
            gimbal_model=gimbal_model,
            data_path=data_path,
            data_dirs=["cam0", "cam1"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv"
        )
        calib.optimize()
