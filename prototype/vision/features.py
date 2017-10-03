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
        """ Constructor

        Args:

            threshold (float): Threshold
            nonmax_supression (bool): Nonmax supression

        """
        # Parameters
        threshold = kwargs.get("threshold", 25)
        nonmax_suppression = kwargs.get("nonmax_supression", True)

        # Detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression
        )

    def detect(self, frame, debug=False):
        keypoints = self.detector.detect(frame)

        if debug is True:
            image = None
            image = cv2.drawKeypoints(frame, keypoints, None)
            cv2.imshow("Keypoints", image)

        return [Keypoint(kp.pt) for kp in keypoints]


class FeatureTrack:
    """ Feature Track """

    def __init__(self, track_id, frame_id, keypoint):
        """ Constructor

        Args:

            frame_id (int): Frame id
            track_id (int): Track id
            keypoint (Keypoint): Keypoint

        """
        self.track_id = track_id
        self.frame_start = frame_id
        self.frame_end = frame_id
        self.track = [keypoint]

    def update(self, frame_id, keypoint):
        """ Update feature track

        Args:

            frame_id (int): Frame id
            keypoint (Keypoint): Keypoint

        """
        self.frame_end = frame_id
        self.track.append(keypoint)

    def last_keypoint(self):
        """ Return last keypoint

        Returns:

            Last keypoint (Keypoint)

        """
        return self.track[-1].pt

    def __str__(self):
        s = ""
        s += "track_id: %d\n" % self.track_id
        s += "frame_start: %d\n" % self.frame_start
        s += "frame_end: %d\n" % self.frame_end
        s += "track: %s" % self.track
        return s


class FeatureTracker:
    """ Feature Tracker """

    def __init__(self, detector):
        """ Constructor

        Args:

            detector (FastDetector): Feature detector

        """
        self.frame_id = 0
        self.detector = detector

        self.track_id = 0
        self.tracks = {}
        self.tracks_alive = []

        self.frame_prev = None
        self.kp_cur = None
        self.kp_ref = None
        self.min_nb_features = 100

    def detect(self, frame):
        """ Detect features

        Detects and initializes feature tracks

        Args:

            frame_id (int): Frame id
            frame (np.array): Frame

        """
        for kp in self.detector.detect(frame):
            track = FeatureTrack(self.track_id, self.frame_id, kp)
            self.tracks[self.track_id] = track
            self.tracks_alive.append(self.track_id)
            self.track_id += 1

    def last_keypoints(self):
        """ Returns previously tracked features

        Returns:

            Keypoints as a numpy array of shape (N, 2), where N is the number of
            keypoints

        """
        keypoints = []
        for track_id in self.tracks_alive:
            keypoints.append(self.tracks[track_id].last_keypoint())

        return np.array(keypoints, dtype=np.float32)

    def track_features(self, image_ref, image_cur):
        """ Track Features

        Args:

            image_ref (np.array): Reference image
            image_cur (np.array): Current image

        """
        # Re-detect new feature points if too few
        if len(self.tracks_alive) < self.min_nb_features:
            self.tracks_alive = []  # reset alive feature tracks
            self.detect(image_ref)

        # LK parameters
        win_size = (21, 21)
        max_level = 2
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)

        # Convert reference keypoints to numpy array
        self.kp_ref = self.last_keypoints()

        # Perform LK tracking
        lk_params = {"winSize": win_size,
                     "maxLevel": max_level,
                     "criteria": criteria}
        self.kp_cur, statuses, err = cv2.calcOpticalFlowPyrLK(image_ref,
                                                              image_cur,
                                                              self.kp_ref,
                                                              None,
                                                              **lk_params)

        # Filter out bad matches (choose only good keypoints)
        status = statuses.reshape(statuses.shape[0])
        still_alive = []
        for i in range(len(status)):
            if status[i] == 1:
                track_id = self.tracks_alive[i]
                still_alive.append(track_id)
                kp = Keypoint(self.kp_cur[i])
                self.tracks[track_id].update(self.frame_id, kp)

        self.tracks_alive = still_alive

    def draw_tracks(self, frame, debug=False):
        """ Draw tracks

        Args:

            frame (np.array): Image frame
            debug (bool): Debug mode

        """
        if debug is False:
            return

        # Create a mask image and color for drawing purposes
        mask = np.zeros_like(frame)
        color = [0, 0, 255]

        # Draw tracks
        for i, (new, old) in enumerate(zip(self.kp_cur, self.kp_ref)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color, 1)
        img = cv2.add(frame, mask)
        cv2.imshow("Feature Tracks", img)

    def update(self, frame, debug=False):
        """ Update

        Args:

            frame (np.array): Image frame
            debug (bool): Debug mode

        """
        if self.frame_id > 0:  # Update
            self.track_features(self.frame_prev, frame)
            self.draw_tracks(frame, debug)

        elif self.frame_id == 0:  # Initialize
            self.detect(frame)

        # Keep track of current frame
        self.frame_prev = frame
        self.frame_id += 1
