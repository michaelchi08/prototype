import cv2
import numpy as np

from prototype.vision.feature2d.feature import Feature
from prototype.vision.feature2d.feature_track import FeatureTrack
from prototype.vision.feature2d.feature_container import FeatureContainer


class KLTTracker:
    """KLT feature tracker

    Attributes
    ---------
    features : FeatureContainer
        Feature container
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
        self.features = FeatureContainer()
        self.counter_frame_id = -1
        self.camera_model = None

        self.nb_max_corners = 1000
        self.quality_level = 0.01
        self.min_distance = 10.0

        self.img_cur = None
        self.img_ref = None

    def initialize(self, img_cur):
        self.img_ref = img_cur
        self.counter_frame_id += 1
        return self.detect(img_cur)

    def get_lost_tracks(self):
        # Get lost tracks
        tracks = self.features.removeLostTracks()
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

    def detect(self, image):
        # Convert image to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Feature detection
        corners = cv2.goodFeaturesToTrack(gray_image,
                                          self.nb_max_corners,
                                          self.quality_level,
                                          self.min_distance)

        # Return features
        features = [Feature(corner) for corner in corners]
        return features

    def process_track(self, status, f0, f1, keeping):
        # Lost - Remove feature track
        if status == 0 and f0.track_id is not None:
            self.features.remove_track(f0.track_id, True)
            return

        # Tracked - Add or update feature track
        if f0.track_id is None:
            self.features.add_track(self.counter_frame_id, f0, f1)
        else:
            self.features.update_track(self.counter_frame_id, f0.track_id, f1)
        keeping.append(f1)

    def track(self, features):
        # Convert list of features to list of cv2.Point2f
        p0 = np.array([feature.pt for feature in features])

        # Convert input images to gray scale
        gray_img_cur = cv2.cvtColor(self.img_cur, cv2.COLOR_BGR2GRAY)
        gray_img_ref = cv2.cvtColor(self.img_ref, cv2.COLOR_BGR2GRAY)

        # Track features
        win_size = (21, 21)
        max_level = 2
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        params = {"winSize": win_size,
                  "maxLevel": max_level,
                  "criteria": criteria}
        p1, status, err = cv2.calcOpticalFlowPyrLK(gray_img_ref,
                                                   gray_img_cur,
                                                   p0,
                                                   None,
                                                   **params)

        # Add, update or remove feature tracks
        keeping = []
        for i in range(len(status)):
            f0 = features[i]
            f1 = Feature(p1[i])
            self.process_track(status[i], f0, f1, keeping)
        self.features.fea_ref = keeping

    def update(self, img_cur):
        # Keep track of current image
        self.img_cur = img_cur

        # Initialize feature tracker
        if len(self.features.fea_ref) == 0:
            self.features.fea_ref = self.initialize(img_cur)
            return

        # Detect
        if (len(self.features.fea_ref) < 200):
            self.features.fea_ref = self.detect(img_cur)
        self.counter_frame_id += 1

        # Track features
        self.track(self.features.fea_ref)
