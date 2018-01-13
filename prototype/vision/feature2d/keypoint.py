import cv2
import numpy as np


class KeyPoint:
    """KeyPoint"""

    def __init__(self,
                 pt,
                 size,
                 angle=-1.0,
                 response=0.0,
                 octave=0,
                 class_id=-1):
        self.pt = np.array(pt)
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id

    def as_cv_keypoint(self):
        return cv2.KeyPoint(self.pt[0],
                            self.pt[1],
                            self.size,
                            self.angle,
                            self.response,
                            self.octave,
                            self.class_id)

    def __str__(self):
        return str(self.pt)
