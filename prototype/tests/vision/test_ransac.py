import os
import random
import unittest

import cv2
import numpy as np
import matplotlib.pylab as plt

import prototype.tests as test
from prototype.vision.ransac import RANSAC
from prototype.vision.ransac import VerticalRANSAC
from prototype.vision.features import FeatureTracker


class RANSACTest(unittest.TestCase):
    def setUp(self):
        self.ransac = RANSAC()

        # Setup test data based on line equation y = mx + c
        self.m_true = 1.0  # Gradient
        self.c_true = 2.0  # Intersection
        x = np.linspace(0.0, 10.0, num=100)
        y = self.m_true * x + self.c_true

        # Add guassian noise to data
        for i in range(len(y)):
            y[i] += np.random.normal(0.0, 0.2)

        # Add outliers to 30% of data
        for i in range(int(len(y) * 0.3)):
            idx = np.random.randint(0, len(y))
            y[idx] += np.random.normal(0.0, 1.0)

        self.data = np.array([x, y])

    def test_sample(self):
        sample = self.ransac.sample(self.data)
        self.assertEqual(2, len(sample))

    def test_distance(self):
        sample = self.ransac.sample(self.data)
        dist = self.ransac.compute_distance(sample, self.data)
        self.assertEqual((1, 100), dist.shape)

    def test_compute_inliers(self):
        sample = self.ransac.sample(self.data)
        dist = self.ransac.compute_distance(sample, self.data)
        self.ransac.compute_inliers(dist)

    # def test_optimize(self):
    #     debug = False
    #
    #     for i in range(10):
    #         m_pred, c_pred, mask = self.ransac.optimize(self.data)
    #         if debug:
    #             print("m_true: ", self.m_true)
    #             print("m_pred: ", m_pred)
    #             print("c_true: ", self.c_true)
    #             print("c_pred: ", c_pred)
    #
    #         self.assertTrue(abs(m_pred - self.m_true) < 0.5)
    #         self.assertTrue(abs(c_pred - self.c_true) < 0.5)
    #
    #         # Plot RANSAC optimized result
    #         debug = False
    #         if debug:
    #             x = np.linspace(0.0, 10.0, num=100)
    #             y = m_pred * x + c_pred
    #             plt.scatter(self.data[0, :], self.data[1, :])
    #             plt.plot(x, y)
    #             plt.show()


class VerticalRANSACTest(unittest.TestCase):
    def setUp(self):
        self.image_height = 600
        self.ransac = VerticalRANSAC(self.image_height)

        # Load test images
        data_path = test.TEST_DATA_PATH
        img0 = cv2.imread(os.path.join(data_path, "vo", "0.png"))
        img1 = cv2.imread(os.path.join(data_path, "vo", "1.png"))

        # Detect features
        tracker = FeatureTracker()
        f0 = tracker.detect(img0)
        f1 = tracker.detect(img1)

        # Convert Features to cv2.KeyPoint and descriptors (np.array)
        kps0 = [cv2.KeyPoint(f.pt[0], f.pt[1], f.size) for f in f0]
        des0 = np.array([f.des for f in f0])
        kps1 = [cv2.KeyPoint(f.pt[0], f.pt[1], f.size) for f in f1]
        des1 = np.array([f.des for f in f1])

        # Perform matching and sort based on distance
        # Note: arguments to the brute-force matcher is (query descriptors,
        # train descriptors), here we use des1 as the query descriptors becase
        # des1 represents the latest descriptors from the latest image frame
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des0)
        matches = sorted(matches, key=lambda x: x.distance)

        # Prepare data for RANSAC outlier rejection
        self.src_pts = np.float32([kps0[m.trainIdx].pt for m in matches])
        self.dst_pts = np.float32([kps1[m.queryIdx].pt for m in matches])

    def test_init(self):
        self.assertEqual(self.ransac.image_height, self.image_height)
        self.assertEqual(self.ransac.max_iter, 100)
        self.assertEqual(self.ransac.threshold, 0.5)
        self.assertEqual(self.ransac.inlier_ratio, 0.2)

    def test_compute_distance(self):
        rand_idx = random.randint(0, self.src_pts.shape[1] - 1)
        sample = np.array([self.src_pts[rand_idx], self.dst_pts[rand_idx]])
        dist = self.ransac.compute_distance(sample, self.src_pts, self.dst_pts)

        self.assertEqual(dist.shape, (self.src_pts.shape[0], 1))

    def test_compute_inliers(self):
        rand_idx = random.randint(0, self.src_pts.shape[1] - 1)
        sample = np.array([self.src_pts[rand_idx], self.dst_pts[rand_idx]])
        dist = self.ransac.compute_distance(sample, self.src_pts, self.dst_pts)
        inliers, nb_inliers = self.ransac.compute_inliers(dist)

        self.assertEqual(inliers.shape, dist.shape)
        self.assertTrue(type(inliers[0, 0]) is np.bool_)

    def test_match(self):
        inlier_indicies = self.ransac.match(self.src_pts, self.dst_pts)
        self.assertEqual(inlier_indicies.shape, (self.src_pts.shape[0], 1))
