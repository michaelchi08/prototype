import numpy as np
from numpy.matlib import repmat


class RANSAC:
    """ RANSAC """

    def __init__(self):
        self.max_iter = 100
        self.threshold = 0.1
        self.inlier_ratio = 0.2

    def sample(self, data):
        """ Random sample 2 points to form a random consensus

        Args:

            data (np.array): Data

        """
        sample = []

        for i in range(2):
            idx = np.random.randint(0, data.shape[1])
            sample.append(data[:, idx])

        return np.array(sample)

    def compute_distance(self, sample, data):
        """ Compute distance between line formed by sample and between data

        Args:

            sample (np.array): Random 2 points for form a random consensus
            data (np.array): Data

        Returns:

            dist (np.array): Vector of distance between data point and line
            formed by 2 random points

        """
        p1 = sample[0].reshape((2, 1))
        p2 = sample[1].reshape((2, 1))

        # Compute the distances between all points with the fitting line
        k_line = p2 - p1  # two points relative distance
        k_line_norm = k_line / np.linalg.norm(k_line)

        # Ax + By + C = 0
        # A = -k_line_norm[1]
        # B = k_line_norm[0]
        v_norm = np.array([-k_line_norm[1], k_line_norm[0]])
        X = repmat(p1, 1, data.shape[1])
        dist = np.dot(v_norm.T, (data - X))

        return dist

    def compute_inliers(self, dist):
        """ Compute inliers

        Args:

            dist (np.array): Distance vector

        Returns:

            (inlier_indicies, nb_inliers): Inlier indicies and number of
            inliers

        """
        inlier_indicies = np.where(abs(dist) <= self.threshold)[1]
        nb_inliers = len(inlier_indicies)
        return (inlier_indicies, nb_inliers)

    def optimize(self, data):
        """ Optimize

        Args:

            data (np.array): Data

        """
        inlier_threshold = round(self.inlier_ratio * data.shape[1])
        best_nb_inliers = 0
        m = 0.0
        c = 0.0

        for i in range(self.max_iter):
            sample = self.sample(data)
            dist = self.compute_distance(sample, data)
            (inlier_indicies, nb_inliers) = self.compute_inliers(dist)

            if nb_inliers >= inlier_threshold and nb_inliers > best_nb_inliers:
                best_nb_inliers = nb_inliers

                p1 = sample[0].reshape((2, 1))
                p2 = sample[1].reshape((2, 1))

                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                c = p1[1] - m * p1[0]

        return (m, c)
