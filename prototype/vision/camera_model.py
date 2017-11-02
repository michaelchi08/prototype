import numpy as np

from prototype.utils.utils import euler2rot
from prototype.vision.common import projection_matrix


class PinholeCameraModel(object):
    def __init__(self, image_width, image_height, K, hz=None):
        """ Constructor

        Args:

            image_width (int): Image width
            image_height (int): Image height
            K (np.array - size 3x3): Camera intrinsics
            hz (int): Frame rate

        """
        self.image_width = image_width
        self.image_height = image_height
        self.hz = hz
        self.K = K

        self.frame = 0
        self.dt = 0.0

    def update(self, dt):
        """Update camera

        Args:

            dt (float): Time difference

        Returns:

            Boolean to denote whether model has been updated

        """
        self.dt += dt

        if self.dt > (1.0 / self.hz):
            self.dt = 0.0
            self.frame += 1
            return True

        return False

    def P(self, R, t):
        return np.dot(self.K, np.block([R, np.dot(-R, t)]))

    def project(self, X, R, t):
        """ Project 3D point to image plane

        Args:

            X (np.array - size 4xN): 3D features in Homogeneous coordinates
            R (np.array - size 3x3): Rotation matrix
            t (np.array - size 3x1): Translation vector

        Returns:

            Projected 3D feature onto 2D image plane

        """
        # Project 3D point to image plane
        P = projection_matrix(self.K, R, np.dot(-R, t))
        x = np.dot(P, X)

        # Normalize pixel coordinates
        x[0] /= x[2]
        x[1] /= x[2]
        x[2] /= x[2]

        return x

    def check_features(self, dt, features, rpy, t):
        """ Check whether features are observable by camera

        Args:

            dt (float): Time difference
            features (np.array): Landmarks
            rpy (np.array or list - size 3): Roll, pitch and yaw
            t (np.array - size 3): Translation vector

        Returns:

            Observed 3D features

        """
        observed = []

        # pre-check
        if self.update(dt) == False:
            return None

        # rotation matrix
        R = euler2rot(rpy, 123)

        # projection matrix
        P = projection_matrix(self.K, R, np.dot(-R, t))

        # check which features are observable from camera
        for i in range(len(features)):
            # convert feature in NWU to EDN coordinate system
            point = features[i]
            point_edn = [0, 0, 0, 0]
            point_edn[0] = -point[1]
            point_edn[1] = -point[2]
            point_edn[2] = point[0]
            point_edn[3] = 1.0
            point_edn = np.array(point_edn)

            # project 3D world point to 2D image plane
            img_pt = np.dot(P, point_edn)

            # check to see if feature is valid and infront of camera
            if img_pt[2] < 1.0:
                continue  # skip this landmark! feature is not infront of camera

            # normalize pixels
            img_pt[0] = img_pt[0] / img_pt[2]
            img_pt[1] = img_pt[1] / img_pt[2]
            img_pt[2] = img_pt[2] / img_pt[2]

            # check to see if feature observed is within image plane
            x_ok = (img_pt[0] < self.image_width) and (img_pt[0] > 0.0)
            y_ok = (img_pt[1] < self.image_height) and (img_pt[1] > 0.0)
            if x_ok and y_ok:
                observed.append((img_pt[0:2], i))

        return observed