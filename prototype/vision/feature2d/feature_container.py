from prototype.vision.feature2d.feature_track import FeatureTrack


class FeatureContainer:
    """Feature Container

    Attributes
    ----------
    counter_track_id : int
        Counter Track ID
    tracks_tracking : :obj`list` of :obj`int`
        List of feature track id
    tracks_lost : :obj`list` of :obj`int`
        List of lost feature track id
    tracks_buffer : :obj`dict` of :obj`FeatureTrack`
        Tracks buffer
    max_buffer_size : int
        Max buffer size (Default: 5000)


    """
    def __init__(self, **kwargs):
        self.counter_track_id = -1

        self.tracking = []
        self.lost = []
        self.data = {}
        self.max_buffer_size = 5000

        self.fea_ref = []

    def add_track(self, frame_id, feature1, feature2):
        """Add feature track

        Parameters
        ----------
        frame_id : int
            Frame id
        feature1 : Feature
            First feature
        feature2 : Feature
            Second feature

        """
        assert frame_id != 0

        self.counter_track_id += 1
        track_id = self.counter_track_id

        feature1.set_track_id(track_id)
        feature2.set_track_id(track_id)

        track = FeatureTrack(track_id, frame_id, feature1, feature2)
        self.tracking.append(track_id)
        self.data[track_id] = track

    def remove_track(self, track_id, lost=False):
        """Remove feature track

        Important! Marking the track as lost does not remove the track from
        the feature track buffer.

        Parameters
        ----------
        track_id : int
            Feature track id
        lost : bool
            Mark feature track as lost

        """
        self.tracking.remove(track_id)
        if lost:
            self.lost.append(track_id)
        else:
            del self.data[track_id]

    def update_track(self, frame_id, track_id, feature):
        """Update feature track

        Parameters
        ----------
        frame_id : int
            Frame id
        track_id : int
            Feature track id
        feature : Feature
            Latest feature

        """
        assert frame_id > 1

        feature.set_track_id(track_id)
        track = self.data[track_id]
        track.update(frame_id, feature)
