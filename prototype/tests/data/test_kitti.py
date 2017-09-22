import os
import unittest


from prototype.data.kitti import VOSequence


# GLOBAL VARIABLES
DATASET_PATH = "/data/dataset"


class VOSequenceTest(unittest.TestCase):
    def test_init(self):
        vo_seq = VOSequence(DATASET_PATH, "00")

        self.assertEqual(4541, len(vo_seq.time))
        self.assertEqual((3, 4), vo_seq.P0.shape)
        self.assertEqual((3, 4), vo_seq.P1.shape)
        self.assertEqual((3, 4), vo_seq.P2.shape)
        self.assertEqual((3, 4), vo_seq.P3.shape)
        self.assertEqual((3, 4), vo_seq.Tr.shape)
