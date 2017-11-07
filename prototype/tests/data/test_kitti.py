import os
import unittest


from prototype.data.kitti import VOSequence
from prototype.data.kitti import RawSequence


# GLOBAL VARIABLES
VO_DATASET = "/data/vo"
RAW_DATASET = "/data/raw"


class VOSequenceTest(unittest.TestCase):
    def test_init(self):
        vo_seq = VOSequence(VO_DATASET, "00")

        self.assertEqual(4541, len(vo_seq.time))
        self.assertEqual((3, 4), vo_seq.P0.shape)
        self.assertEqual((3, 4), vo_seq.P1.shape)
        self.assertEqual((3, 4), vo_seq.P2.shape)
        self.assertEqual((3, 4), vo_seq.P3.shape)


class RawSequenceTest(unittest.TestCase):
    def test_init(self):
        RawSequence(RAW_DATASET, "2011_09_26", "0001")
