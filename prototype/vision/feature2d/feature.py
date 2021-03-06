import numpy as np


class Feature:
    """Feature"""

    def __init__(self, pt, size=0, des=None):
        """ Constructor

        Args:

            pt (np.array): Point
            size (float): Size
            des (np.array): Descriptor

        """
        self.pt = pt if type(pt) == np.ndarray else np.array(pt)
        self.size = size
        self.des = des

        self.track_id = None

    def set_track_id(self, track_id):
        self.track_id = track_id

    def __str__(self):
        return str(self.pt)
