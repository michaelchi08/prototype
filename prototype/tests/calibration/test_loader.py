import unittest
from os.path import join

import prototype.tests as test
from prototype.calibration.loader import DataLoader


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.data_path = join(test.TEST_DATA_PATH, "calib_data")

    def test_load_imu_data(self):
        loader = DataLoader(
            data_path=self.data_path,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv",
            nb_rows=6,
            nb_cols=7,
            square_size=0.29
        )

        joint_data = loader.load_joint_data()
        self.assertEqual(5, joint_data.shape[0])
        self.assertEqual(3, joint_data.shape[1])

    def test_load(self):
        loader = DataLoader(
            data_path=self.data_path,
            inspect_data=True,
            image_dirs=["static_camera", "gimbal_camera"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv",
            nb_rows=6,
            nb_cols=7,
            square_size=0.29
        )
        loader.load()

    def test_load_preprocessed(self):
        data_path = "/home/chutsu/Dropbox/calib_data/"
        loader = DataLoader(
            preprocessed=True,
            data_path=data_path,
            data_dirs=["cam0", "cam1"],
            intrinsic_files=["static_camera.yaml", "gimbal_camera.yaml"],
            joint_file="joint.csv"
        )
        loader.load()
