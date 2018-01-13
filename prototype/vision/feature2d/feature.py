import numpy as np


class Feature:
    """Feature"""

    def __init__(self, pt, size, des):
        """ Constructor

        Args:

            pt (np.array): Point
            size (float): Size
            des (np.array): Descriptor

        """
        self.pt = np.array(pt)
        self.size = size
        self.des = des

        self.track_id = None

    def set_track_id(self, track_id):
        self.track_id = track_id

    def __str__(self):
        return str(self.pt)
