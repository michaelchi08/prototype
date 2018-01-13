import cv2
import numpy as np

from prototype.vision.feature2d.fast import FAST
from prototype.vision.feature2d.keypoint import KeyPoint
from prototype.vision.feature2d.feature_track import FeatureTrack


class LKFeatureTracker:
    """Lucas-Kanade Feature Tracker"""

    def __init__(self, **kwargs):
        """ Constructor

        Args:

            detector : FAST
                Feature detector

        """
        self.frame_id = 0
        self.detector = kwargs.get("detector", FAST(threshold=2))

        self.track_id = 0
        self.tracks = {}
        self.tracks_tracking = []

        self.frame_prev = None
        self.kp_cur = None
        self.kp_ref = None
        self.min_nb_features = 100

    def detect(self, frame):
        """Detect features

        Detects and initializes feature tracks

        Parameters
        ----------
        frame_id : int
            Frame id
        frame : np.array
            Frame

        """
        for kp in self.detector.detect_keypoints(frame):
            track = FeatureTrack(self.track_id, self.frame_id, kp)
            self.tracks[self.track_id] = track
            self.tracks_tracking.append(self.track_id)
            self.track_id += 1

    def last_keypoints(self):
        """Returns previously tracked features

        Returns
        -------

            KeyPoints as a numpy array of shape (N, 2), where N is the number
            of keypoints

        """
        keypoints = []
        for track_id in self.tracks_tracking:
            keypoints.append(self.tracks[track_id].last().pt)

        return np.array(keypoints, dtype=np.float32)

    def track_features(self, image_ref, image_cur):
        """Track Features

        Parameters
        ----------
        image_ref : np.array
            Reference image
        image_cur : np.array
            Current image

        """
        # Re-detect new feature points if too few
        if len(self.tracks_tracking) < self.min_nb_features:
            self.tracks_tracking = []  # reset alive feature tracks
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
                track_id = self.tracks_tracking[i]
                still_alive.append(track_id)
                kp = KeyPoint(self.kp_cur[i], 0)
                self.tracks[track_id].update(self.frame_id, kp)

        self.tracks_tracking = still_alive

    def draw_tracks(self, frame, debug=False):
        """Draw tracks

        Parameters
        ----------
        frame : np.array
            Image frame
        debug : bool
            Debug mode (Default value = False)

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
        """Update

        Parameters
        ----------
        frame : np.array
            Image frame
        debug : bool
            Debug mode (Default value = False)

        Returns
        -------

        """
        if self.frame_id > 0:  # Update
            self.track_features(self.frame_prev, frame)
            self.draw_tracks(frame, debug)

        elif self.frame_id == 0:  # Initialize
            self.detect(frame)

        # Keep track of current frame
        self.frame_prev = frame
        self.frame_id += 1
