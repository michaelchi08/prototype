import math

import cv2
import numpy as np


class Keypoint:
    """ Keypoint """

    def __init__(self, pt, size):
        self.pt = np.array(pt)
        self.size = size
        self.angle = None
        self.response = None
        self.octave = None

    def __str__(self):
        return str(self.pt)


class Keyframe:
    """ Keyframe """

    def __init__(self, image, features):
        """ Constructor

        Args:

            image (np.array): Image
            features (List of Features): Features

        """
        self.image = image
        self.features = features

    def update(self, image, features):
        self.image = image
        self.features = features


class Feature:
    """ Feature """
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

    def __str__(self):
        return str(self.pt)


class FeatureTrack:
    """ Feature Track """

    def __init__(self, track_id, frame_id, data):
        """ Constructor

        Args:

            frame_id (int): Frame id
            track_id (int): Track id
            data (Keypoint or Feature): data

        """
        self.track_id = track_id
        self.frame_start = frame_id
        self.frame_end = frame_id
        self.track = [data]

    def update(self, frame_id, data):
        """ Update feature track

        Args:

            frame_id (int): Frame id
            data (Keypoint or Feature): data

        """
        self.frame_end = frame_id
        self.track.append(data)

    def last(self):
        """ Return last data point

        Returns:

            Last keypoint (Keypoint)

        """
        return self.track[-1]

    def last_descriptor(self):
        """ Return last descriptor

        Returns:

            Last descriptor (Descriptor)

        """
        return self.descriptor[-1]

    def __str__(self):
        s = ""
        s += "track_id: %d\n" % self.track_id
        s += "frame_start: %d\n" % self.frame_start
        s += "frame_end: %d\n" % self.frame_end
        s += "track: %s" % self.track
        return s


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

        return [Keypoint(kp.pt, kp.size) for kp in keypoints]


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

            features

        List of Features

        """
        # Detect and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)

        # Convert OpenCV keypoints and descriptors to Features
        features = []
        for i in range(len(keypoints)):
            kp = keypoints[i]
            dp = descriptors[i]
            features.append(Feature(kp.pt, kp.size, dp))

        # Show debug image
        if debug is True:
            image = None
            image = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
            cv2.imshow("Keypoints", image)

        return features


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
            keypoints.append(self.tracks[track_id].last().pt)

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
                kp = Keypoint(self.kp_cur[i], 0)
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
        self.detector = ORBDetector(nfeatures=1000, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.track_id = 0
        self.tracks_buffer = {}
        self.tracks_alive = []
        self.max_buffer_size = math.inf

        self.keyframe = None
        self.feature_threshold = 0.4

    def detect(self, frame):
        """ Detect features

        Detects and initializes feature tracks

        Args:

            frame (np.array): Frame

        """
        return self.detector.detect(frame)

    def match(self, tracks, features):
        """ Match features to feature tracks

        The idea is that with the current features, we want to match it against
        the current list of FeatureTrack.

        Args:

            tracks (List of FeatureTrack): Current feature tracks
            features (List of Features): Current features

        Returns:

            (feature track id, feature index)

        """
        # Convert Features to cv2.KeyPoint and descriptors (np.array)
        kps1 = []
        des1 = []
        for t in tracks:
            f = t.last()
            kps1.append(cv2.KeyPoint(f.pt[0], f.pt[1], f.size))
            des1.append(f.des)
        des1 = np.array(des1)

        kps2 = [cv2.KeyPoint(f.pt[0], f.pt[1], f.size) for f in features]
        des2 = np.array([f.des for f in features])

        # Perform matching and sort based on distance
        # Note: arguments to the brute-force matcher is (query descriptors,
        # train descriptors), here we use des2 as the query descriptors becase
        # des2 represents the latest descriptors from the latest image frame
        matches = self.matcher.match(des2, des1)
        matches = sorted(matches, key=lambda x: x.distance)

        # Perform RANSAC (by utilizing findFundamentalMat()) on matches
        # This acts as a stage 2 filter where outliers are rejected
        src_pts = np.float32([kps1[m.trainIdx].pt for m in matches])
        dst_pts = np.float32([kps2[m.queryIdx].pt for m in matches])
        src_pts = src_pts.reshape(-1, 1, 2)
        dst_pts = dst_pts.reshape(-1, 1, 2)
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts)
        match_mask = mask.ravel().tolist()

        # Remove outliers
        final_matches = []
        for i in range(len(match_mask)):
            if match_mask[i] == 1:
                final_matches.append(matches[i])

        # Return matches in the form of (feature track id, feature index)
        result = []
        for m in final_matches:
            match = (tracks[m.trainIdx].track_id, m.queryIdx)
            result.append(match)

        return result

    def draw_matches(self,
                     img_ref, img_cur,
                     tracks, features,
                     matches):
        """ Draw matches

        NOTE: Call this before calling FeatureTracker.update_feature_tracks(),
        else the feature track last keypoint will be the same as the current
        feature keypoint matched.

        Args:

            img_ref (np.array): Reference image
            img_cur (np.array): Current image
            tracks (List of FeatureTrack): Feature tracks currently tracking
            features (List of Features): Current features
            matches (Tuple of float): (feature track id, feature index)

        """
        # Vertically stack images with latest frame on top
        img = np.vstack((img_cur, img_ref))

        # Draw matches
        for m in matches:
            track_id, feature_idx = m
            kp1 = self.tracks_buffer[track_id].track[0].pt
            kp2 = features[feature_idx].pt

            # Point 1
            p1 = np.array(kp1)
            p1[1] += img.shape[0] / 2.0
            p1 = (int(p1[0]), int(p1[1]))

            # Point 2
            p2 = np.array(kp2)
            p2 = (int(p2[0]), int(p2[1]))

            cv2.line(img, p2, p1, (0, 255, 0), 1)

        return img

    def init_feature_tracks(self, features):
        """ Initialize feature tracks

        Args:

            features (List of Features): Features

        """
        # Make sure tracks alive is clear
        self.tracks_alive.clear()

        # Initialize each feature with its own FeatureTrack
        for f in features:
            track = FeatureTrack(self.track_id, self.frame_id, f)
            self.tracks_buffer[self.track_id] = track
            self.tracks_alive.append(track)
            self.track_id += 1

    def update_feature_tracks(self, features, matches):
        """ Update feature tracks

        Args:

            features (List of Features): Current features
            matches (Tuple of float): (feature track id, feature index)

        """
        # Update feature tracks currently tracking
        tracks_updated = {}
        for m in matches:
            track_id, feature_idx = m
            track = self.tracks_buffer[track_id]
            feature = features[feature_idx]
            track.update(self.frame_id, feature)
            tracks_updated[track_id] = 1

        # Drop dead feature tracks
        tracks_still_alive = []
        for t in self.tracks_alive:
            if t.track_id in tracks_updated:
                tracks_still_alive.append(t)
        self.tracks_alive = list(tracks_still_alive)

        # Clear feature tracks that are too old
        # if len(self.tracks_buffer) > self.max_buffer_size:
        #     self.tracks_buffer = self.tracks_buffer[500:-1]

    def update(self, img_cur):
        """ Update tracker with current image

        Args:

            img_cur (np.array): Current image

        """
        # Initialize tracker
        if self.keyframe is None:
            features = self.detect(img_cur)
            self.keyframe = Keyframe(img_cur, features)
            self.init_feature_tracks(self.keyframe.features)
            self.frame_id += 1
            return

        # Match features
        features = self.detect(img_cur)
        matches = self.match(self.tracks_alive, features)

        # Draw matches
        img = self.draw_matches(self.keyframe.image, img_cur,
                                self.tracks_alive, features,
                                matches)
        cv2.imshow("Matches", img)

        # Update
        self.update_feature_tracks(features, matches)
        if len(matches) < len(self.keyframe.features) * self.feature_threshold:
            self.keyframe.update(img_cur, features)
            self.init_feature_tracks(self.keyframe.features)
        self.frame_id += 1
