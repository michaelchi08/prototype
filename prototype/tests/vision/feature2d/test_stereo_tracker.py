import unittest
from os.path import join

import cv2
import numpy as np
import matplotlib.pylab as plt

from prototype.data.kitti import RawSequence
from prototype.vision.feature2d.common import draw_points
from prototype.vision.feature2d.common import draw_features
from prototype.vision.feature2d.common import draw_matches
from prototype.vision.feature2d.stereo_tracker import StereoTracker
from prototype.vision.feature2d.feature_container import FeatureContainer


class StereoTrackerTest(unittest.TestCase):
    def setUp(self):
        # Load KITTI data
        data_path = join("/data", "kitti", "raw")
        self.data = RawSequence(data_path, "2011_09_26", "0005")

        # Stereo tracker
        self.tracker = StereoTracker()

    def test_detect(self):
        img = cv2.imread(self.data.image_00_files[0])
        features = self.tracker.detect(img, 1000)
        self.assertTrue(len(features) > 0)

        # Draw keypoints
        # debug = True
        debug = False
        if debug:
            img = draw_features(img, features)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

    def test_track(self):
        img_ref = cv2.imread(self.data.image_00_files[0])
        img_cur = cv2.imread(self.data.image_00_files[1])
        fea_ref = self.tracker.detect(img_ref, 1000)
        buf = FeatureContainer()

        keeping = self.tracker.track(img_ref, img_cur, buf, fea_ref)
        self.assertTrue(len(keeping) > 0)

    def test_process_track(self):
        img0 = cv2.imread(self.data.image_00_files[0])
        img1 = cv2.imread(self.data.image_01_files[0])
        self.tracker.initialize(img0, img1)

        # Test add track
        self.tracker.counter_frame_id += 1
        f0 = self.tracker.fea0_ref[0]
        f1 = self.tracker.fea0_ref[1]
        self.tracker.process_track(1, f0, f1, self.tracker.buf0)
        self.assertEqual(1, len(self.tracker.buf0.tracking))
        self.assertEqual(0, len(self.tracker.buf0.lost))
        self.assertEqual(1, len(self.tracker.buf0.data))

        # Test update track
        self.tracker.counter_frame_id += 1
        f2 = self.tracker.fea0_ref[2]
        f0 = self.tracker.buf0.data[0]
        self.tracker.process_track(1, f0, f2, self.tracker.buf0)
        self.assertEqual(1, len(self.tracker.buf0.tracking))
        self.assertEqual(0, len(self.tracker.buf0.lost))
        self.assertEqual(1, len(self.tracker.buf0.data))
        self.assertEqual(3, len(f0.track))

        # Test remove track
        self.tracker.counter_frame_id += 1
        self.tracker.process_track(0, f0, f2, self.tracker.buf0)
        self.assertEqual(0, len(self.tracker.buf0.tracking))
        self.assertEqual(1, len(self.tracker.buf0.lost))
        self.assertEqual(1, len(self.tracker.buf0.data))

    def test_temporal_match(self):
        img_ref = cv2.imread(self.data.image_00_files[0])
        img_cur = cv2.imread(self.data.image_00_files[1])

        fea_ref = self.tracker.detect(img_ref, 1000)
        fea_ref, fea_cur = self.tracker.track(img_ref, img_cur,
                                              self.tracker.buf0, fea_ref)
        inlier_mask = self.tracker.temporal_match(fea_ref, fea_cur)

        # Plot matches
        # debug = True
        debug = False
        if debug:
            match_img = draw_matches(img_ref, img_cur,
                                     fea_ref, fea_cur,
                                     inlier_mask)
            cv2.imshow("Matches", match_img)
            cv2.waitKey(0)

    # def test_grid_sampling(self):
    #     img = cv2.imread(self.data.image_00_files[0])
    #     fea = self.tracker.detect(img)
    #
    #     image_width = img.shape[1]
    #     image_height = img.shape[0]
    #
    #     pts = np.array([f.pt for f in fea])
    #
    #     grid = []
    #     divisions = 10
    #     for x in range(0, image_width, int(image_width / divisions)):
    #         for y in range(0, image_height, int(image_height / divisions)):
    #             grid.append([x, y])
    #     grid = np.array(grid)

        # plt.scatter(x, y, marker="o", color="red")
        # plt.scatter(pts[:, 0], pts[:, 1], marker="o")
        # plt.xlim([0, image_width])
        # plt.ylim([0, image_height])
        # plt.gca().invert_yaxis()
        # plt.show()

    def test_update(self):
        # Get number of images and image size
        nb_images = len(self.data.image_00_files)
        img_size = cv2.imread(self.data.image_00_files[0]).shape
        img_size = (int(img_size[1] / 2), int(img_size[0] / 2))

        # Loop through images
        for i in range(nb_images):
            # Load image and resize image
            img0 = cv2.imread(self.data.image_00_files[i])
            img1 = cv2.imread(self.data.image_01_files[i])
            img0 = cv2.resize(img0, img_size)
            img1 = cv2.resize(img1, img_size)

            # # Detect features
            # self.tracker.update(img0, img1)

            # # # Show images
            # img0 = draw_features(img0, self.tracker.fea0_ref)
            # img1 = draw_features(img1, self.tracker.fea1_ref)
            # cv2.imshow("Image", np.hstack((img0, img1)))
            cv2.imshow("Image", img0)
            # print(img0)
            if cv2.waitKey(0) == 113:
                break
