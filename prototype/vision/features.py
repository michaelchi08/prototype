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
        # Parameters
        threshold = kwargs.get("threshold", 25)
        nonmax_suppression = kwargs.get("nonmax_supression", True)

        # Detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression
        )

    def detect(self, frame):
        keypoints = self.detector.detect(frame)

        image = None
        image = cv2.drawKeypoints(frame, keypoints, None)
        cv2.imshow("Keypoints", image)

        return [Keypoint(kp.pt) for kp in keypoints]


class FeatureTrack:
    """ Feature Track """

    def __init__(self, track_id, frame_id, keypoint):
        self.track_id = 0
        self.frame_start = frame_id
        self.frame_end = frame_id
        self.track = [keypoint]


class FeatureTracker:
    """ Feature Tracker """

    def __init__(self, detector):
        self.frame_id = 0
        self.detector = detector
        self.feature_tracks = []

        self.frame_prev = None
        self.kp_cur = None
        self.kp_ref = None
        self.min_nb_features = 100

    def track_features(self, image_ref, image_cur, kp_old):
        """ Track Features

        Args:

            image_ref (np.array): Reference image
            image_cur (np.array): Current image
            kp_old (list of Keypoints): Old keypoints

        Returns:

            Old and new keypoints

        """
        # Re-detect new feature points if too few
        if len(kp_old) < self.min_nb_features:
            kp_old = self.detector.detect(image_ref)

        # LK parameters
        win_size = (21, 21)
        max_level = 2
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)

        # Convert reference keypoints to numpy array
        kp_old = np.array([x.pt for x in kp_old], dtype=np.float32)

        # Perform LK tracking
        lk_params = {"winSize": win_size,
                     "maxLevel": max_level,
                     "criteria": criteria}
        kp_new, st, err = cv2.calcOpticalFlowPyrLK(image_ref,
                                                   image_cur,
                                                   kp_old,
                                                   None,
                                                   **lk_params)

        # Post-process (choose only good keypoints)
        st = st.reshape(st.shape[0])
        kp_old = kp_old[st == 1]
        kp_new = kp_new[st == 1]

        # Convert np.array back to keypoints
        kp_old = [Keypoint(x) for x in kp_old]
        kp_new = [Keypoint(x) for x in kp_new]

        return kp_old, kp_new

    def draw_tracks(self, kp1, kp2, frame, debug=False):
        if debug is False:
            return

        # Create a mask image and color for drawing purposes
        mask = np.zeros_like(frame)
        color = [0, 0, 255]

        # Convert Keypoints to numpy array
        kp1 = np.array([x.pt for x in kp1], dtype=np.float32)
        kp2 = np.array([x.pt for x in kp2], dtype=np.float32)

        # Draw tracks
        for i, (new, old) in enumerate(zip(kp2, kp1)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color, 1)
        img = cv2.add(frame, mask)
        cv2.imshow("Feature Tracks", img)

    def update(self, frame, debug=False):
        if self.frame_id > 0:
            # Track features
            self.kp_ref, self.kp_cur = self.track_features(self.frame_prev,
                                                           frame,
                                                           self.kp_ref)
            self.draw_tracks(self.kp_ref, self.kp_cur, frame, debug)

        elif self.frame_id == 0:
            # First frame
            self.kp_ref = self.detector.detect(frame)

        # Keep track of current frame
        self.kp_ref = self.kp_cur if self.kp_cur else self.kp_ref
        self.frame_prev = frame
        self.frame_id += 1
