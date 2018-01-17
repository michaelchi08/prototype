import numpy as np


class FeatureTrack:
    """Feature Track

    Attributes
    ----------
    track_id : int
        Track id
    frame_start : int
        First frame where feature was observed
    frame_end : int
        Last frame where feature was observed
    track : list of Feature
        List of features representing the feature track
    ground_truth : int or np.array - 3x1
        Feature position ground truth

    Parameters
    ----------
    track_id : int
        Track id
    frame_id : int
        Frame id
    data0 : KeyPoint or Feature
        First KeyPoint or feature
    data1 : KeyPoint or Feature
        Second KeyPoint or feature
    ground_truth : int or np.array - 3x1
        Feature position ground truth

    """
    def __init__(self, track_id, frame_id, data0, data1=None, **kwargs):
        self.track_id = track_id

        if data1 is None:
            self.frame_start = frame_id
            self.frame_end = frame_id
            self.track = [data0]
        else:
            self.frame_start = frame_id - 1
            self.frame_end = frame_id
            self.track = [data0, data1]

        self.ground_truth = kwargs.get("ground_truth", None)
        self.pos = kwargs.get("pos", [])
        self.rpy = kwargs.get("rpy", [])

    def update(self, frame_id, data, pos=None, rpy=None):
        """Update feature track

        Parameters
        ----------
        frame_id : int
            Frame id
        data : KeyPoint or Feature
            data

        """
        self.frame_end = frame_id
        self.track.append(data)
        if pos is not None:
            self.pos.append(pos)
        if rpy is not None:
            self.rpy.append(rpy)

    def last(self):
        """Return last data point

        Returns
        -------
        data_point : KeyPoint or Feature
            Last data point

        """
        data_point = self.track[-1]
        return data_point

    def tracked_length(self):
        """Return number of frames tracked

        Returns
        -------
        length : float
            Number of frames tracked

        """
        length = len(self.track)
        return length

    def __str__(self):
        s = ""
        s += "track_id: %d\n" % self.track_id
        s += "frame_start: %d\n" % self.frame_start
        s += "frame_end: %d\n" % self.frame_end
        s += "track: \n"
        for t in self.track:
            s += "\t%.2f, %.2f" % (np.round(t.pt[0], 2), np.round(t.pt[1], 2))
            s += "\n"
        s += "\n"
        if self.ground_truth is not None:
            s += "\n"
            s += "ground_truth: " + str(self.ground_truth)
            s += "\n"
        return s
