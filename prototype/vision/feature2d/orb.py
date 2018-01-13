import cv2

from prototype.vision.feature2d.keypoint import KeyPoint
from prototype.vision.feature2d.feature import Feature
from prototype.vision.feature2d.common import convert2cvkeypoints


class ORB:
    """ORB"""

    def __init__(self, **kwargs):
        """ Constructor """
        self.orb = cv2.ORB_create(
            nfeatures=kwargs.get("nfeatures", 500),
            scaleFactor=kwargs.get("scaleFactor", 1.2),
            nlevels=kwargs.get("nlevels", 8),
            edgeThreshold=kwargs.get("edgeThreshold", 31),
            firstLevel=kwargs.get("firstLevel", 0),
            WTA_K=kwargs.get("WTA_K", 2),
            scoreType=kwargs.get("scoreType", cv2.ORB_HARRIS_SCORE),
            patchSize=kwargs.get("patchSize", 31),
            fastThreshold=kwargs.get("fastThreshold", 20)
        )

    def detect_keypoints(self, frame):
        """Detect keypoints

        Parameters
        ----------
        frame : np.array
            Image frame

        Returns
        -------
        kps : list of KeyPoint
            Keypoints

        """
        # Detect keypoints
        keypoints = self.orb.detect(frame, None)

        # Convert OpenCV KeyPoint to KeyPoint
        kps = []
        for i in range(len(keypoints)):
            kp = keypoints[i]
            kps.append(KeyPoint(kp.pt,
                                kp.size,
                                kp.angle,
                                kp.response,
                                kp.octave))

        return kps

    def detect_features(self, frame):
        """Detect features (Detector + Descriptor)

        Parameters
        ----------
        frame : np.array
            Image frame
        debug : bool
            Debug mode (Default value = False)

        Returns
        -------
        features : list of Feature
            Features

        """
        # Detect and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)

        # Convert OpenCV keypoints and descriptors to Features
        features = []
        for i in range(len(keypoints)):
            kp = keypoints[i]
            dp = descriptors[i]
            features.append(Feature(kp.pt, kp.size, dp))

        return features

    def extract_descriptors(self, frame, kps):
        """Extract feature descriptors

        Parameters
        ----------
        frame : np.array
            Image frame
        kps: list of KeyPoint
            Key points

        Returns
        -------
        des : list of np.array
            Descriptors

        """
        cv_kps = convert2cvkeypoints(kps)
        cv_kps, des = self.orb.compute(frame, cv_kps)

        # Convert OpenCV KeyPoint to KeyPoint
        kps = []
        for cv_kp in cv_kps:
            kps.append(KeyPoint(cv_kp.pt,
                                cv_kp.size,
                                cv_kp.angle,
                                cv_kp.response,
                                cv_kp.octave,
                                cv_kp.class_id))

        return kps, des
