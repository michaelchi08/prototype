#!/usr/bin/env python3
import cv2
import numpy as np

STAGE_DEFAULT_FRAME = 0
STAGE_FIRST_FRAME = 1
STAGE_SECOND_FRAME = 2
kMinNumFeature = 1500


def featureTracking(image_ref, image_cur, px_ref):
    lk_params = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    }

    # shape: [k,2] [k,1] [k,1]
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref,
                                            image_cur,
                                            px_ref,
                                            None,
                                            **lk_params)

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class BasicVO:
    """ """

    def __init__(self, cam, annotations):
        self.frame_id = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25,
                                                       nonmaxSuppression=True)

        with open(annotations) as f:
            self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id):
        ss = self.annotations[frame_id - 1].strip().split()

        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])

        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])

        self.trueX = x
        self.trueY = y
        self.trueZ = z

        dx = (x - x_prev)
        dy = (y - y_prev)
        dz = (z - z_prev)

        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref],
                               dtype=np.float32)
        self.frame_id = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(
            self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(
            self.px_cur, self.px_ref, focal=self.focal, pp=self.pp,
            method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(
            E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
        self.frame_id = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking(self.last_frame,
                                                        self.new_frame,
                                                        self.px_ref)

        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)

        _, R, t, mask = cv2.recoverPose(E,
                                        self.px_cur,
                                        self.px_ref,
                                        focal=self.focal,
                                        pp=self.pp)

        scale = self.getAbsoluteScale(frame_id)
        if scale > 0.1:
            self.cur_t = self.cur_t + scale*self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)

        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array(
                [x.pt for x in self.px_cur],
                dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert(img.ndim == 2, "Invalid image shape, image is not 2D")
        assert(img.shape[0] == self.cam.height, "Invalid image height")
        assert(img.shape[1] == self.cam.width, "Invalud image width!")

        # update
        self.new_frame = img
        if self.frame_id == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_id == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_id == STAGE_FIRST_FRAME:
            self.processFirstFrame()

        # keep track of current frame
        self.last_frame = self.new_frame
