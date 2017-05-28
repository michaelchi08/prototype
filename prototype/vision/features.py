#!/usr/bin/env python3
import cv2
import numpy as np


class Keypoint(object):
    def __init__(self, pt):
        self.pt = np.array(pt)

    def __str__(self):
        return str(self.pt)


class FastDetector(object):
    def __init__(self, **kwargs):
        # parameters
        threshold = kwargs.get("threshold", 25)
        nonmax_suppression = kwargs.get("nonmax_supression", True)

        # detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression
        )

    def detect(self, frame):
        keypoints = self.detector.detect(frame)
        return [Keypoint(kp.pt) for kp in keypoints]


def vo():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
