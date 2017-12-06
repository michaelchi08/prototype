import os
import unittest


import prototype.tests as test
from prototype.data.kitti import VOSequence
from prototype.data.kitti import RawSequence


class VOSequenceTest(unittest.TestCase):
    def test_init(self):
        data_path = os.path.join(test.TEST_DATA_PATH, "kitti", "vo")
        vo_seq = VOSequence(data_path, "00")

        self.assertEqual(4541, len(vo_seq.time))
        self.assertEqual((3, 4), vo_seq.P0.shape)
        self.assertEqual((3, 4), vo_seq.P1.shape)
        self.assertEqual((3, 4), vo_seq.P2.shape)
        self.assertEqual((3, 4), vo_seq.P3.shape)


class RawSequenceTest(unittest.TestCase):
    def test_init(self):
        data_path = os.path.join(test.TEST_DATA_PATH, "kitti", "raw")
        RawSequence(data_path, "2011_09_26", "0001")
