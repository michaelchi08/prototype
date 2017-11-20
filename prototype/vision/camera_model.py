import numpy as np

from prototype.utils.euler import euler2rot
from prototype.vision.common import projection_matrix
from prototype.vision.common import convert2homogeneous




class PinholeCameraModel:
    """ Pinhole camera model

    Attributes
    ----------
    image_width: int
        Image width
    image_height: int
        Image height
    K : np.array - 3x3
        Camera intrinsics

    Parameters
    ----------
    image_width : int
        Image width
    image_height : int
        Image height
    K : np.array - 3x3
        Camera intrinsics

    """

    def __init__(self, image_width, image_height, K):
        self.image_width = image_width
        self.image_height = image_height
        self.K = K

    def P(self, R, t):
        """Return projection matrix

        Parameters
        ----------
        R : np.array - 3x3
            Rotation matrix
        t : np.array - 3x1
            Translation vector

        Returns
        -------
        P : np.array - 3x4
            Projection matrix

        """
        t = t.reshape((3, 1))
        P = np.dot(self.K, np.block([R, np.dot(-R, t)]))
        return P

    def project(self, X, R, t):
        """Project 3D point to image plane

        Parameters
        ----------
        X : np.array - 3xN
            3D features in camera coordinates
        R : np.array - 3x3
            Rotation matrix
        t : np.array - 3x1
            Translation vector

        Returns
        -------
        x : np.array
            Projected 3D feature onto 2D image plane

        """
        # Correct the input, has to be of shape 3xN where N is nb features
        if len(X.shape) == 1:
            X = X.reshape((3, int(len(X) / 3)))

        # Convert 3D features to homogenous coordinates
        X = convert2homogeneous(X)

        # Project 3D point to image plane
        P = projection_matrix(self.K, R, np.dot(-R, t))
        x = np.dot(P, X)

        # Normalize pixel coordinates
        x[0] /= x[2]
        x[1] /= x[2]
        x[2] /= x[2]
        x = np.array(x)

        return x

    def pixel2image(self, pixel):
        """Convert pixel measurement to image coordinates

        Parameters
        ----------
        pixel : np.array - 2x1
            Pixel measurement

        Returns
        -------
        pt : np.array - 2x1
            Pixel in image coordinates

        """
        cx, cy = self.K[0, 2], self.K[1, 2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        pt = np.array([(pixel[0] - cx) / fx, (pixel[1] - cy) / fy])
        return pt

    def observed_features(self, features, rpy, t):
        """Return features are observed by camera

        Parameters
        ----------
        features : np.array
            Features
        rpy : np.array or list - size 3
            Roll, pitch and yaw
        t : np.array - size 3
            Translation vector

        Returns
        -------
        observed : list of [[np.array - 1x2, int], ...]
            List of observed 3D features as a tuple (pixel coordinates, feature
            idx)

        """
        observed = []

        # rotation matrix
        R = euler2rot(rpy, 123)

        # projection matrix
        P = projection_matrix(self.K, R, np.dot(-R, t))

        # check which features are observable from camera
        feature_idx = 0
        for f in features.T:
            # project 3D world point to 2D image plane
            point = np.array([f[0], f[1], f[2], 1.0])
            img_pt = np.dot(P, point)

            # check to see if feature is valid and infront of camera
            if img_pt[2] < 1.0:
                continue  # skip this feature! It is not infront of camera

            # normalize pixels
            img_pt[0] = img_pt[0] / img_pt[2]
            img_pt[1] = img_pt[1] / img_pt[2]
            img_pt[2] = img_pt[2] / img_pt[2]

            # check to see if feature observed is within image plane
            x_ok = (img_pt[0] < self.image_width) and (img_pt[0] > 0.0)
            y_ok = (img_pt[1] < self.image_height) and (img_pt[1] > 0.0)
            if x_ok and y_ok:
                observed.append((img_pt[0:2], feature_idx))

            # Update
            feature_idx += 1

        return observed
