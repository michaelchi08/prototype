import cv2
import numpy as np

from prototype.vision.feature2d.feature import Feature
from prototype.vision.feature2d.feature_container import FeatureContainer


class StereoTracker:
    """Stereo KLT feature tracker

    Attributes
    ---------
    features : FeatureContainer
        Feature Container
    counter_frame_id : int
        Frame id counter
    camera_model : None
        Camera model

    nb_max_corners : int
        Max number of corners
    quality_level : float
        Quality level
    min_distance : float
        Min distance

    img_cur : np.array
        Current image
    img_ref : np.array
        Reference image

    """
    def __init__(self):
        self.buf0 = FeatureContainer()
        self.buf1 = FeatureContainer()
        self.counter_frame_id = -1
        self.camera_model = None

        self.nb_max_corners = 1000
        self.quality_level = 0.01
        self.min_distance = 10.0

        self.img0_ref = None
        self.img1_ref = None

        self.fea0_ref = None
        self.fea1_ref = None

        # KLT settings
        self.win_size = (21, 21)
        self.max_level = 2
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03) # NOQA

    def get_lost_tracks(self):
        # Get lost tracks
        tracks = self.buf0.removeLostTracks()
        if self.camera_model is None:
            return tracks

        # Transform keypoints
        for track in tracks:
            for feature in track.track:
                # Convert pixel coordinates to image coordinates
                pt = self.camera_model.pixel2image(feature.kp.pt)
                feature.kp.pt.x = pt(0)
                feature.kp.pt.y = pt(1)

        return tracks

    def initialize(self, img0_cur, img1_cur):
        self.fea0_ref = self.detect(img0_cur)
        self.fea1_ref = self.detect(img1_cur)
        self.counter_frame_id += 1
        self.img0_ref = img0_cur
        self.img1_ref = img1_cur

    def detect(self, image):
        # Convert image to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Feature detection with Good Features To Track
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray_image, None)
        corners = np.array([[[kp.pt[0], kp.pt[1]]] for kp in keypoints])
        corners = corners.astype(np.float32)

        # Return features
        features = [Feature(corner[0]) for corner in corners]
        return features

    def process_track(self, status, fref, fcur, buf):
        # Lost - Remove feature track
        if status == 0 and fref.track_id is not None:
            buf.remove_track(fref.track_id, True)
            return False

        # Tracked - Add or update feature track
        track_id = fref.track_id
        if track_id is None:
            buf.add_track(self.counter_frame_id, fref, fcur)
        else:
            buf.update_track(self.counter_frame_id, track_id, fcur)
        return True

    def track(self, img_ref, img_cur, buf, fea_ref):
        # Convert list of features to list of cv2.Point2f
        p0 = np.array([f.pt for f in fea_ref])

        # Convert input images to gray scale
        gray_img_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY)
        gray_img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

        # Track features
        params = {"winSize": self.win_size,
                  "maxLevel": self.max_level,
                  "criteria": self.criteria}
        p1, status, err = cv2.calcOpticalFlowPyrLK(gray_img_ref,
                                                   gray_img_cur,
                                                   p0,
                                                   None,
                                                   **params)

        # Add, update or remove feature tracks
        tracked_fea_ref = []
        tracked_fea_cur = []
        for i in range(len(status)):
            fref = fea_ref[i]
            fcur = Feature(p1[i])
            if self.process_track(status[i], fref, fcur, buf):
                tracked_fea_ref.append(fref)
                tracked_fea_cur.append(fcur)

        return (tracked_fea_ref, tracked_fea_cur)

    def temporal_match(self, fea_ref, fea_cur):
        src = np.array([f.pt for f in fea_ref])
        dst = np.array([f.pt for f in fea_cur])
        F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC)

        inliers = []
        for i in range(len(inlier_mask)):
            if inlier_mask[i]:
                inliers.append(fea_cur[i])

        return inliers

    # TODO: use R, t to find points within next image and perform ransac
    def stereo_match(self, fea0, fea1):
        src = np.array([f.pt for f in fea0])
        dst = np.array([f.pt for f in fea1])
        F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC)

        inliers_fea0 = []
        inliers_fea1 = []
        for i in range(len(inlier_mask)):
            if inlier_mask[i]:
                inliers_fea0.append(fea0[i])
                inliers_fea1.append(fea1[i])

        return inliers_fea0, inliers_fea1

    def update(self, img0_cur, img1_cur):
        # Initialize feature tracker
        if self.fea0_ref is None and self.fea1_ref is None:
            self.initialize(img0_cur, img1_cur)
            return

        # Track features
        self.counter_frame_id += 1
        fea0_ref, fea0_cur = self.track(self.img0_ref, img0_cur,
                                        self.buf0, self.fea0_ref)
        fea1_ref, fea1_cur = self.track(self.img1_ref, img1_cur,
                                        self.buf1, self.fea1_ref)

        # Match features
        self.fea0_ref = self.temporal_match(fea0_ref, fea0_cur)
        self.fea1_ref = self.temporal_match(fea1_ref, fea1_cur)

        # Keep track of current image
        self.img0_ref = img0_cur
        self.img1_ref = img1_cur
