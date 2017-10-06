import cv2
import sys
import numpy as np


class Keypoint:
    """ Keypoint """

    def __init__(self, pt):
        self.pt = np.array(pt)

    def __str__(self):
        return str(self.pt)


class FASTDetector:
    """ Fast Detector """

    def __init__(self, **kwargs):
        """ Constructor

        Args:

            threshold (float): Threshold
            nonmax_supression (bool): Nonmax supression

        """
        self.detector = cv2.FastFeatureDetector_create(
            threshold=kwargs.get("threshold", 25),
            nonmaxSuppression=kwargs.get("nonmax_supression", True)
        )

    def detect(self, frame, debug=False):
        """ Detect

        Args:

            frame (np.array): Image frame
            debug (bool): Debug mode

        Returns:

            List of Keypoints

        """
        # Detect
        keypoints = self.detector.detect(frame)

        # Show debug image
        if debug is True:
            image = None
            image = cv2.drawKeypoints(frame, keypoints, None)
            cv2.imshow("Keypoints", image)

        return [Keypoint(kp.pt) for kp in keypoints]


class ORBDetector:
    """ ORB Detector """

    def __init__(self, **kwargs):
        """ Constructor """
        self.detector = cv2.ORB_create(
            nfeatures=kwargs.get("nfeatures", 500),
            scaleFactor=kwargs.get("scaleFactor", 1.2),
            nlevels=kwargs.get("nlevels", 8),
            edgeThreshold=kwargs.get("edgeThreshold", 31),
            firstLevel=kwargs.get("firstLevel", 0),
            WTA_K=kwargs.get("WTA_K", 2),
            scoreType=kwargs.get("scoreType", cv2.ORB_HARRIS_SCORE),
            patchSize=kwargs.get("patchSize", 31),
            fastThreshold=kwargs.get("fastThreshold", 20)
        )

    def detect(self, frame, debug=False):
        """ Detect

        Args:

            frame (np.array): Image frame
            debug (bool): Debug mode

        Returns:

            (keypoints, descriptors)

        Tuple of list of keypoints and numpy array of descriptors

        """
        # Detect and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)

        # Show debug image
        if debug is True:
            image = None
            image = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
            cv2.imshow("Keypoints", image)

        return [Keypoint(kp.pt) for kp in keypoints], descriptors


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


class LKFeatureTracker:
    """ Lucas-Kanade Feature Tracker """

    def __init__(self, detector):
        """ Constructor

        Args:

            detector (FASTDetector): Feature detector

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


class FeatureTracker:
    """ Feature Tracker """

    def __init__(self):
        """ Constructor """
        self.frame_id = 0
        self.detector = ORBDetector(nfeatures=500, nlevels=8)

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

    def draw_matches(self, img_ref, img_cur, kps1, kps2, matches, match_mask):
        # Vertically stack images with latest on top
        img = np.vstack((img_cur, img_ref))

        # Draw matches
        for i in range(len(matches)):
            if match_mask[i]:
                p1 = list(kps1[matches[i].queryIdx].pt)
                p1[1] += img.shape[0] / 2.0
                p1 = (int(p1[0]), int(p1[1]))

                p2 = list(kps2[matches[i].trainIdx].pt)
                p2 = (int(p2[0]), int(p2[1]))

                cv2.line(img, p2, p1, (0, 255, 0), 1)

        # Show matches
        cv2.imshow("Matches", img)

    def match(self, img_ref, img_cur):
        kps1, des1 = self.detector.detector.detectAndCompute(img_ref, None)
        kps2, des2 = self.detector.detector.detectAndCompute(img_cur, None)

        # Setup matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Perform matching and sort based on distance
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Perform RANSAC (by utilizing findFundamentalMat()) on matches
        match_mask = None
        if len(matches) > 10:
            src_pts = np.float32([kps1[m.queryIdx].pt for m in matches])
            src_pts = src_pts.reshape(-1, 1, 2)
            dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches])
            dst_pts = dst_pts.reshape(-1, 1, 2)

            M, mask = cv2.findFundamentalMat(src_pts, dst_pts)
            match_mask = mask.ravel().tolist()

        else:
            print("Not enough matches! - ({}/{})".format(len(matches), 10))

        # Draw matches
        self.draw_matches(img_ref, img_cur, kps1, kps2, matches, match_mask)
