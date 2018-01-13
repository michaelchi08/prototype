import cv2

from prototype.vision.feature2d.keypoint import KeyPoint


class FAST:
    """FAST Detector

    Parameters
    ----------
    threshold : float
        Threshold

    nonmax_suppression : bool
        Nonmax supression

    """

    def __init__(self, **kwargs):
        self.detector = cv2.FastFeatureDetector_create(
            threshold=kwargs.get("threshold", 25),
            nonmaxSuppression=kwargs.get("nonmax_suppression", True)
        )

    def detect_keypoints(self, frame):
        """Detect

        Parameters
        ----------
        frame : np.array
            Image frame
        debug : bool
            Debug mode (Default value = False)

        Returns
        -------
        results : list of KeyPoint
            List of KeyPoints

        """
        # Detect
        keypoints = self.detector.detect(frame)
        results = [KeyPoint(kp.pt, kp.size) for kp in keypoints]

        return results
