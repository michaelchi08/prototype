import cv2
import numpy as np

from prototype.vision.ransac import VerticalRANSAC


class KeyPoint:
    """KeyPoint"""

    def __init__(self, pt, size):
        self.pt = np.array(pt)
        self.size = size
        self.angle = None
        self.response = None
        self.octave = None

    def __str__(self):
        return str(self.pt)


class Keyframe:
    """Keyframe"""

    def __init__(self, image, features):
        """ Constructor

        Args:

            image (np.array): Image
            features (List of Features): Features

        """
        self.image = image
        self.features = features

    def update(self, image, features):
        """

        Parameters
        ----------
        image :

        features :


        Returns
        -------

        """
        self.image = image
        self.features = features


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

    def update(self, frame_id, data):
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

    def last(self):
        """Return last data point

        Parameters
        ----------

        Returns
        -------
        type
            Last keypoint (KeyPoint)

        """
        return self.track[-1]

    def tracked_length(self):
        """Return number of frames tracked

        Parameters
        ----------

        Returns
        -------
        type
            Number of frames tracked (float)

        """
        return len(self.track)

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


class FAST:
    """FAST Detector

    Parameters
    ----------
    threshold : float
        Threshold

    nonmax_suppression : bool
        Nonmax supression

    """

    def __init__(self, **kwargs):
        self.detector = cv2.FastFeatureDetector_create(
            threshold=kwargs.get("threshold", 25),
            nonmaxSuppression=kwargs.get("nonmax_suppression", True)
        )

    def detect(self, frame, debug=False):
        """Detect

        Parameters
        ----------
        frame : np.array
            Image frame
        debug : bool
            Debug mode (Default value = False)

        Returns
        -------
        results : list of KeyPoint
            List of KeyPoints

        """
        # Detect
        keypoints = self.detector.detect(frame)

        # Show debug image
        if debug is True:
            image = None
            image = cv2.drawKeyPoints(frame, keypoints, None)
            cv2.imshow("KeyPoints", image)

        results = [KeyPoint(kp.pt, kp.size) for kp in keypoints]

        return results


class ORB:
    """ORB"""

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
        """Detect

        Parameters
        ----------
        frame : np.array
            Image frame
        debug : bool
            Debug mode (Default value = False)

        Returns
        -------

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
            image = cv2.drawKeyPoints(
                frame, keypoints, None, color=(0, 255, 0))
            cv2.imshow("KeyPoints", image)

        return features


class LKFeatureTracker:
    """Lucas-Kanade Feature Tracker"""

    def __init__(self, detector):
        """ Constructor

        Args:

            detector : FAST
                Feature detector

        """
        self.frame_id = 0
        self.detector = detector

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
        for kp in self.detector.detect(frame):
            track = FeatureTrack(self.track_id, self.frame_id, kp)
            self.tracks[self.track_id] = track
            self.tracks_tracking.append(self.track_id)
            self.track_id += 1

    def last_keypoints(self):
        """Returns previously tracked features

        Parameters
        ----------

        Returns
        -------

            KeyPoints as a numpy array of shape (N, 2), where N is the number of
            keypoints

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


class FeatureTracker:
    """Feature Tracker

    Attributes
    ----------
    detector :
        Feature detector
    matcher :
        Feature matcher

    counter_frame_id : int
        Counter Frame ID
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

    img_ref : np.array
        Reference image
    fea_ref :
        Reference feature
    unmatched : :obj`list` of `Feature`
        List of features

    """

    def __init__(self, **kwargs):
        self.debug_mode = kwargs.get("debug_mode", False)
        self.nb_features = kwargs.get("nb_features", 500)
        self.nb_levels = kwargs.get("nb_levels", 4)

        # Detector and matcher
        self.detector = ORB(nfeatures=self.nb_features,
                            nlevels=self.nb_levels)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ransac = None

        # Counters
        self.counter_frame_id = -1
        self.counter_track_id = -1

        # Feature tracks
        self.tracks_tracking = []
        self.tracks_lost = []
        self.tracks_buffer = {}
        self.max_buffer_size = 5000

        # Image, feature, unmatched features references
        self.img_ref = None
        self.fea_ref = None
        self.unmatched = []

    def debug(self, s):
        if self.debug_mode:
            print(s)

    def add_track(self, feature1, feature2):
        """Add feature track

        Parameters
        ----------
        feature1 : Feature
            First feature
        feature2 : Feature
            Second feature

        """
        self.counter_track_id += 1
        track_id = self.counter_track_id
        frame_id = self.counter_frame_id

        feature1.set_track_id(track_id)
        feature2.set_track_id(track_id)

        track = FeatureTrack(track_id, frame_id, feature1, feature2)
        self.tracks_tracking.append(track_id)
        self.tracks_buffer[track_id] = track

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
        self.tracks_tracking.remove(track_id)
        if lost:
            self.tracks_lost.append(track_id)
        else:
            del self.tracks_buffer[track_id]

    def update_track(self, track_id, feature):
        """Update feature track

        Parameters
        ----------
        track_id : int
            Feature track id
        feature : Feature
            Latest feature

        """
        feature.set_track_id(track_id)
        track = self.tracks_buffer[track_id]
        track.update(self.counter_frame_id, feature)

    def initialize(self, frame):
        """Initialize feature tracker

        Parameters
        ----------
        frame : np.array
            Frame

        """
        self.img_ref = frame
        self.fea_ref = self.detect(frame)

        if self.ransac is None:
            self.ransac = VerticalRANSAC(frame.shape[1])

    def detect(self, frame):
        """Detect features

        Detects and initializes feature tracks

        Parameters
        ----------
        frame : np.array
            Frame

        Returns
        -------
        features : :obj`list` of :obj`Feature`

        """
        features = self.detector.detect(frame)
        self.counter_frame_id += 1
        return features

    def match(self, f1):
        """Match features to feature tracks

        The idea is that with the current features, we want to match it against
        the current list of FeatureTrack.

        Parameters
        ----------
        f1 : List of Feature
            Current features

        Returns
        -------
        result :
            (feature track id, f0 index, f1 index)

        """
        # Stack previously unmatched features with tracked features
        f0 = self.fea_ref + self.unmatched

        # Convert Features to cv2.KeyPoint and descriptors (np.array)
        kps0 = [cv2.KeyPoint(f.pt[0], f.pt[1], f.size) for f in f0]
        des0 = np.array([f.des for f in f0])
        kps1 = [cv2.KeyPoint(f.pt[0], f.pt[1], f.size) for f in f1]
        des1 = np.array([f.des for f in f1])

        # Perform matching and sort based on distance
        # Note: arguments to the brute-force matcher is (query descriptors,
        # train descriptors), here we use des1 as the query descriptors becase
        # des1 represents the latest descriptors from the latest image frame
        matches = self.matcher.match(des1, des0)
        matches = sorted(matches, key=lambda x: x.distance)

        # Perform RANSAC (by utilizing findFundamentalMat()) on matches
        # This acts as a stage 2 filter where outliers are rejected
        src_pts = np.float32([kps0[m.trainIdx].pt for m in matches])
        dst_pts = np.float32([kps1[m.queryIdx].pt for m in matches])

        # src_pts = src_pts.reshape(-1, 1, 2)
        # dst_pts = dst_pts.reshape(-1, 1, 2)
        # M, mask = cv2.findFundamentalMat(src_pts, dst_pts,
        #                                  cv2.FM_RANSAC, 1, 0.99)
        # match_mask = mask.ravel().tolist()

        mask = self.ransac.match(src_pts, dst_pts)
        match_mask = mask.ravel().tolist()

        # Remove outliers
        final_matches = []
        for i in range(len(match_mask)):
            # if match_mask[i] == 1:
            if match_mask[i] is True:
                final_matches.append(matches[i])

        # Convert matches to the form of (f0 index, f1 index) and update
        # feature references
        result = []
        self.fea_ref = []
        matched_indicies = {}
        for m in final_matches:
            f0_idx = m.trainIdx
            f1_idx = m.queryIdx
            result.append((f0_idx, f1_idx))
            matched_indicies[m.queryIdx] = True
            self.fea_ref.append(f1[f1_idx])

        # Update list of unmatched features
        self.unmatched.clear()
        for i in range(len(f1)):
            if i not in matched_indicies:
                self.unmatched.append(f1[i])

        return result, f0, f1

    def process_matches(self, matches, f0, f1):
        """Process matches

        Parameters
        ----------
        matches : Tuple of float
            (feature track id, feature index)
        f0 : List of Features
            Reference features
        f1 : List of Features
            Current features

        """
        tracks_updated = {}

        # Update or add feature track
        for i in range(len(matches)):
            f0_idx, f1_idx = matches[i]
            feature0 = f0[f0_idx]
            feature1 = f1[f1_idx]

            if feature0.track_id is not None:
                self.update_track(feature0.track_id, feature1)
                self.debug("Update track [%d]" % feature0.track_id)
            else:
                self.add_track(feature0, feature1)
                self.debug("Add track [%d]" % feature0.track_id)

            tracks_updated[feature0.track_id] = 1

        # Drop dead feature tracks
        tracks_tracking = list(self.tracks_tracking)
        for i in range(len(self.tracks_tracking)):
            track_id = tracks_tracking[i]
            if track_id not in tracks_updated:
                self.remove_track(track_id, True)

        self.debug("Tracking: " + str(self.tracks_tracking))
        self.debug("Lost: " + str(self.tracks_lost))
        self.debug("Buffer: " + str(self.tracks_buffer))
        self.debug("")

    def clear_old_tracks(self):
        """Clear old feature tracks"""
        # Clear feature tracks that are too old
        if len(self.tracks_buffer) > self.max_buffer_size:
            trim = len(self.tracks_buffer) - self.max_buffer_size
            for i in range(trim):
                track_id = self.tracks_lost.pop(0)
                del self.tracks_buffer[track_id]

    def draw_matches(self, img_ref, img_cur, f0, f1, matches):
        """Draw matches

        Parameters
        ----------
        img_ref : np.array
            Reference image
        img_cur : np.array
            Current image
        f0 : :obj`list` of :obj`Feature`
            Reference features
        f1 : :obj`list` of :obj`Feature`
            Current features
        matches : :obj`tuple` of :obj`float`
            (feature track id, f0 index, f1 index)

        Returns
        -------
        match_img : np.array
            Match image

        """
        # Vertically stack images with latest frame on top
        match_img = np.vstack((img_cur, img_ref))

        # Draw matches
        for m in matches:
            f0_idx, f1_idx = m
            track_id = f0[f0_idx].track_id
            kp0 = f0[f0_idx].pt
            kp1 = f1[f1_idx].pt

            # Point 1
            p1 = np.array(kp0)
            img_height = match_img.shape[0]
            p1[1] += img_height / 2.0
            p1 = (int(p1[0]), int(p1[1]))
            match_img = cv2.circle(match_img, p1, 2, (0, 255, 0), -1)

            # Point 2
            p2 = np.array(kp1)
            p2 = (int(p2[0]), int(p2[1]))
            match_img = cv2.circle(match_img, p2, 2, (0, 255, 0), -1)

            # Draw line
            cv2.line(match_img, p2, p1, (0, 255, 0), 1)

            # Draw track id
            if track_id is not None:
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.4
                font_color = (0, 255, 0)

                x = p1[0] + 10
                padding = np.random.randint(-100, 100)
                y = int(p1[1] - img_height / 4.0 + padding)
                pos = (x, y)

                cv2.putText(match_img,
                            str(track_id),
                            pos,
                            font,
                            font_scale,
                            font_color)

        return match_img

    def remove_lost_tracks(self):
        """Remove lost tracks"""
        lost_tracks = []

        # Remove tracks from self.tracks_buffer
        for track_id in self.tracks_lost:
            track = self.tracks_buffer[track_id]
            lost_tracks.append(track)
            del self.tracks_buffer[track_id]

        # Reset tracks lost array
        self.tracks_lost = []

        return lost_tracks

    def update(self, img_cur, show_matches=False):
        """Update tracker with current image

        Parameters
        ----------
        img_cur : np.array
            Current image
        debug : bool
            Debug mode (Default value = False)

        """
        # Initialize tracker
        if self.fea_ref is None:
            self.initialize(img_cur)
            return

        # Detect features on latest image frame
        fea_cur = self.detect(img_cur)

        # Match features
        matches, fea_ref, fea_cur = self.match(fea_cur)
        self.process_matches(matches, fea_ref, fea_cur)

        # Draw matches
        if show_matches:
            img = self.draw_matches(self.img_ref, img_cur,
                                    fea_ref, fea_cur,
                                    matches)
            cv2.imshow("Matches", img)

        # Update
        self.img_ref = img_cur
