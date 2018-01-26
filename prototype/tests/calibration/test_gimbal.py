import unittest
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import prototype.tests as test
from prototype.calibration.chessboard import Chessboard
from prototype.calibration.gimbal import GECDataLoader
from prototype.calibration.gimbal import GEC
from prototype.viz.plot_gimbal import PlotGimbal
from prototype.viz.plot_chessboard import PlotChessboard
from prototype.viz.common import axis_equal_3dplot


class GECDataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.data_loader = GECDataLoader(
            data_path=self.data_path,
            cam0_dir="static_camera",
            cam1_dir="gimbal_camera",
            intrinsics_filename="intrinsics_equi.yaml",
            imu_filename="imu.dat",
            nb_rows=6,
            nb_cols=7,
            square_size=0.29
        )

    def test_load_imu_data(self):
        imu_fpath = join(self.data_path, self.data_loader.imu_filename)
        imu_data = self.data_loader.load_imu_data(imu_fpath)
        self.assertEqual(5, imu_data.shape[0])
        self.assertEqual(3, imu_data.shape[1])

    def test_load_cam_intrinsics(self):
        intrinsics_fpath = join(self.data_path, "intrinsics_equi.yaml")
        self.data_loader.load_cam_intrinsics(intrinsics_fpath)
        self.assertTrue(self.data_loader.data.cam0_intrinsics is not None)
        self.assertTrue(self.data_loader.data.cam1_intrinsics is not None)

    def test_load(self):
        retval = self.data_loader.load()
        self.assertTrue(retval)


class GECTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")
        self.calib = GEC(
            data_path=self.data_path,
            cam0_dir="static_camera",
            cam1_dir="gimbal_camera",
            intrinsics_filename="intrinsics_equi.yaml",
            imu_filename="imu.dat",
            nb_rows=6,
            nb_cols=7,
            square_size=0.0285
        )

    def test_setup_problem(self):
        x, Z = self.calib.setup_problem()

        self.assertEqual((28, ), x.shape)
        self.assertEqual(5, len(Z))

    def test_reprojection_error(self):
        x, Z = self.calib.setup_problem()

        K_s = self.calib.data.cam0_intrinsics.K_new
        K_d = self.calib.data.cam1_intrinsics.K_new
        args = [Z, K_s, K_d]

        self.calib.reprojection_error(x, *args)

    # def test_optimize(self):
    #     data_path = "/home/chutsu/Dropbox/calib_data/extrinsics"
    #     calib = GEC(
    #         data_path=data_path,
    #         cam0_dir="static_camera",
    #         cam1_dir="gimbal_camera",
    #         intrinsics_filename="intrinsics.yaml",
    #         imu_filename="gimbal_joint.dat",
    #         nb_rows=6,
    #         nb_cols=7,
    #         square_size=0.0285
    #     )
    #     calib.optimize()
#
#     def test_sandbox(self):
#         chessboard = Chessboard(t_G=np.array([5.0, 0.0, 1.0]))
#         plot_chessboard = PlotChessboard(chessboard=chessboard)
#         plot_chessboard.plot()
#         plt.show()
