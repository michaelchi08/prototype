import random
import math

import numpy as np
from numpy.matlib import repmat


class RANSAC:
    """RANSAC

    Attributes
    ----------
    max_iter : int
        Max iterations

    threshold : float
        Threshold

    inlier_ratio : float
        Inlier ratio

    """

    def __init__(self):
        self.max_iter = 100
        self.threshold = 0.1
        self.inlier_ratio = 0.2

    def sample(self, data):
        """Random sample 2 points to form a random consensus

        Parameters
        ----------
        data : np.array
            Data

        Returns
        -------

        """
        sample = []

        for i in range(2):
            idx = np.random.randint(0, data.shape[1])
            sample.append(data[:, idx])

        return np.array(sample)

    def compute_distance(self, sample, data):
        """Compute distance between line formed by sample and between data

        Parameters
        ----------
        sample : np.array
            Random 2 points for form a random consensus
        data : np.array
            Data

        Returns
        -------
        dist : np.array
            Vector of distance between data point and line

        """
        p1 = sample[0].reshape((2, 1))
        p2 = sample[1].reshape((2, 1))

        # Calculate unit-norm vector formed by p1 to p2
        k_line = p2 - p1
        if np.linalg.norm(k_line) < 1e-10:
            return None
        k_line_norm = k_line / np.linalg.norm(k_line)

        # Compute the distances between all points with the fitting line
        # Ax + By + C = 0
        # A = -k_line_norm[1]
        # B = k_line_norm[0]
        v_norm = np.array([-k_line_norm[1], k_line_norm[0]])
        X = repmat(p1, 1, data.shape[1])
        dist = np.dot(v_norm.T, (data - X))

        return dist

    def compute_inliers(self, dist):
        """Compute inliers

        Parameters
        ----------
        dist : np.array
            Distance vector

        Returns
        -------
        inlier_indicies : np.array
            Inlier indicies
        nb_inliers : int
            Number of inliers

        """
        inlier_indicies = np.where(abs(dist) <= self.threshold)[1]
        nb_inliers = len(inlier_indicies)
        return (inlier_indicies, nb_inliers)

    def optimize(self, data):
        """Optimize

        Parameters
        ----------
        data : np.array
            Data

        Returns
        -------
        m : float
            Line gradient
        c : float
            Line constant
        inliner_indicies : list of int
            Inlier indicies

        """
        # Setup
        m = 0.0
        c = 0.0
        best_nb_inliers = 0
        inlier_threshold = round(self.inlier_ratio * data.shape[1])

        # Optimize
        for i in range(self.max_iter):
            # Sample and compute point distance
            sample = self.sample(data)
            dist = self.compute_distance(sample, data)
            if dist is None:
                continue

            # Compute inliers
            (inlier_indicies, nb_inliers) = self.compute_inliers(dist)
            if nb_inliers >= inlier_threshold and nb_inliers > best_nb_inliers:
                best_nb_inliers = nb_inliers

                p1 = sample[0].reshape((2, 1))
                p2 = sample[1].reshape((2, 1))

                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                c = p1[1] - m * p1[0]

        return (m, c, inlier_indicies)


class VerticalRANSAC:
    """VerticalRANSAC

    This is a hacked version of the standard RANSAC, this version is useful for
    applications such as visual odometry or SLAM where feature matching outlier
    rejection is needed. The idea in this Vertical RANSAC is the image pair is
    stacked vertically rather than side by side, this has the benefit that if
    the camera motion is moving forwards the feature pair gradients will mostly
    fall within the same gradient direction rather than having +ve or -ve
    gradients with features on the sides of the image.

    Attributes
    ----------
    image_height : int
        Image height in pixels
    max_iter : int
        Max iterations
    threshold : float
        Threshold in degrees
    inlier_ratio : float
        Inlier ratio

    """

    def __init__(self, image_height, **kwargs):
        self.image_height = image_height
        self.max_iter = kwargs.get("max_iter", 100)
        self.threshold = kwargs.get("threshold", 0.5)
        self.inlier_ratio = kwargs.get("inlier_ratio", 0.2)

    def compute_inliers(self, dist):
        """Compute inliers

        Parameters
        ----------
        dist : np.array
            Distance vector

        Returns
        -------
        inlier_indicies
            Inlier indicies and number of
        inlier_indicies
            Inlier indicies and number of
            inliers

        """
        inlier_indicies = np.abs(dist) <= self.threshold
        nb_inliers = len(inlier_indicies)
        return (inlier_indicies, nb_inliers)

    def compute_distance(self, sample, src_pts, dst_pts):
        """Compute angle formed between pairs of feature matches and the
        angle distance between sample and data

        Parameters
        ----------
        sample : np.array
            Random feature match pair to form a random consensus
        src_pts : np.array
            From keypoints
        dst_pts : np.array
            To keypoints

        Returns
        -------
        dist : np.array
            Vector of distance between data point and line

        """
        # Calculate angle formed by sample feature match along the y-axis (not
        # x-axis)
        p1 = sample[0].reshape((2, 1))
        p2 = sample[1].reshape((2, 1)) + np.array([[0.0], [self.image_height]])
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        sample_angle = math.degrees(math.atan2(dx, dy))
        sample_angle = repmat(sample_angle, src_pts.shape[0], 1)

        # Calculate angle formed by all other feature matches along the y-axis
        # (not x-axis)
        data_angles = []
        for i in range(src_pts.shape[0]):
            src_pt = src_pts[i]
            dst_pt = dst_pts[i] + np.array([0.0, self.image_height])
            dx = dst_pt[0] - src_pt[0]
            dy = dst_pt[1] - src_pt[1]
            line_angle = math.degrees(math.atan2(dx, dy))
            data_angles.append(line_angle)
        data_angles = np.array(data_angles).reshape((-1, 1))

        # Calculate distance between sample and data
        dist = sample_angle - data_angles
        return dist

    def match(self, src_pts, dst_pts):
        """Match

        Parameters
        ----------
        src_pts: np.array
            Points 1
        dst_pts: np.array
            Points 2

        Returns
        -------
        inliner_indicies : list of int
            Inlier indicies

        """
        # Setup
        best_nb_inliers = 0
        inlier_threshold = round(self.inlier_ratio * src_pts.shape[1])

        # Optimize
        for i in range(self.max_iter):
            # Sample a feature match pair
            rand_idx = random.randint(0, src_pts.shape[1] - 1)
            sample = np.array([src_pts[rand_idx], dst_pts[rand_idx]])

            # Compute point distance
            dist = self.compute_distance(sample, src_pts, dst_pts)
            if dist is None:
                continue

            # Compute inliers
            (inlier_indicies, nb_inliers) = self.compute_inliers(dist)
            if nb_inliers >= inlier_threshold and nb_inliers > best_nb_inliers:
                best_nb_inliers = nb_inliers

        return inlier_indicies
