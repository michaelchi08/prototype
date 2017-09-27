import cv2
import numpy as np


class Keypoint(object):
    """ Keypoint """

    def __init__(self, pt):
        self.pt = np.array(pt)

    def __str__(self):
        return str(self.pt)


class FastDetector(object):
    """ Fast Detector """

    def __init__(self, **kwargs):
        # Parameters
        threshold = kwargs.get("threshold", 25)
        nonmax_suppression = kwargs.get("nonmax_supression", True)

        # Detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression
        )

    def detect(self, frame):
        keypoints = self.detector.detect(frame)
        return [Keypoint(kp.pt) for kp in keypoints]


class FeatureTrack:
    """ Feature Track """

    def __init__(self, track_id, frame_id, keypoint):
        self.track_id = 0
        self.frame_start = frame_id
        self.frame_end = frame_id
        self.track = [keypoint]


class FeatureTracker:
    """ Feature Tracker """

    def __init__(self, detector):
        self.frame_id = 0
        self.detector = detector
        self.feature_tracks = []

    def detect(self, frame):
        # Detect keypoints
        keypoints = self.detector.detect(frame)

        # Add keypoints to feature tracks
        for kp in keypoints:
            self.feature_tracks.append(FeatureTrack(kp))

        # Update
        self.frame_id += 1
