import cv2
import numpy as np


def feature_tracking(image_ref, image_cur, px_ref):
    """

    Parameters
    ----------
    image_ref :

    image_cur :

    px_ref :


    Returns
    -------

    """
    # Setup
    win_size = (21, 21)
    max_level = 3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Perform LK-tracking
    lk_params = {"winSize": win_size,
                 "maxLevel": max_level,
                 "criteria": criteria}
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref,
                                            image_cur,
                                            px_ref,
                                            None,
                                            **lk_params)

    # Post-process
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class BasicVO:
    """ """

    def __init__(self, fx, cx, cy):
        self.last_frame = None
        self.R = None
        self.t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = fx
        self.pp = (cx, cy)
        self.detector = cv2.FastFeatureDetector_create(threshold=25,
                                                       nonmaxSuppression=True)

    def calc_pose(self, frame):
        """

        Parameters
        ----------
        frame :


        Returns
        -------

        """
        px_ref, px_cur = feature_tracking(self.last_frame, frame, self.px_ref)
        E, mask = cv2.findEssentialMat(px_cur,
                                       px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E,
                                        px_cur,
                                        px_ref,
                                        focal=self.focal,
                                        pp=self.pp)
        return R, t, px_ref, px_cur

    def process(self, frame_id, frame, scale, min_nb_features=1500):
        """

        Parameters
        ----------
        frame_id :

        frame :

        scale :

        min_nb_features :
             (Default value = 1500)

        Returns
        -------

        """
        R, t, self.px_ref, self.px_cur = self.calc_pose(frame)

        # Update position and orientation of camera
        if scale > 0.1:
            self.t = self.t + scale * self.R.dot(t)
            self.R = R.dot(self.R)

        # Re-detect new feature points (too few)
        if self.px_ref.shape[0] < min_nb_features:
            self.px_cur = self.detector.detect(frame)
            self.px_cur = np.array([x.pt for x in self.px_cur],
                                   dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, frame_id, frame, scale):
        """

        Parameters
        ----------
        frame_id :

        frame :

        scale :


        Returns
        -------

        """
        # Update
        if frame_id > 1:
            self.process(frame_id, frame, scale)

        elif frame_id == 1:
            # Second frame
            self.R, self.t, self.px_ref, self.px_cur = self.calc_pose(frame)
            self.px_ref = self.px_cur

        elif frame_id == 0:
            # First frame
            self.px_ref = self.detector.detect(frame)
            self.px_ref = np.array([x.pt for x in self.px_ref],
                                   dtype=np.float32)

        # Keep track of current frame
        self.last_frame = frame

        return (self.R, self.t)
