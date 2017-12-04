import unittest
import ctypes
from ctypes.util import find_library

import cv2
import numpy as np

from prototype.vision.camera import Camera
from prototype.apriltag.apriltag_structs import _ImageU8
from prototype.apriltag.apriltag_structs import _ApriltagFamily
from prototype.apriltag.apriltag_structs import _ApriltagDetector
from prototype.apriltag.apriltag_structs import _ZArray
from prototype.apriltag.apriltag_structs import _ApriltagDetection


def _ptr_to_array2d(datatype, ptr, rows, cols):
    array_type = (datatype * cols) * rows
    array_buf = array_type.from_address(ctypes.addressof(ptr))
    return np.ctypeslib.as_array(array_buf, shape=(rows, cols))


def _image_u8_get_array(img_ptr):
    return _ptr_to_array2d(ctypes.c_uint8,
                           img_ptr.contents.buf.contents,
                           img_ptr.contents.height,
                           img_ptr.contents.stride)


def _matd_get_array(mat_ptr):
    return _ptr_to_array2d(ctypes.c_double,
                           mat_ptr.contents.data,
                           int(mat_ptr.contents.nrows),
                           int(mat_ptr.contents.ncols))


def convert_image(libc, img):
    height = img.shape[0]
    width = img.shape[1]

    libc.image_u8_create.restype = ctypes.POINTER(_ImageU8)
    c_img = libc.image_u8_create(width, height)

    tmp = _image_u8_get_array(c_img)

    # Copy the opencv image into the destination array, accounting for the
    # difference between stride & width.
    tmp[:, :width] = img

    # tmp goes out of scope here but we don't care because
    # the underlying data is still in c_img.
    return c_img


class ApriltagTest(unittest.TestCase):
    def setUp(self):
        libpath = "/usr/local/lib/libapriltag.so"
        self.lib = ctypes.CDLL(libpath)

    def test_image_struct(self):
        # Tag family
        self.lib.tag36h11_create.restype = ctypes.POINTER(_ApriltagFamily)
        family = self.lib.tag36h11_create()
        # self.lib_tag36h10_create()
        # self.lib_tag36artoolkit_create()
        # self.lib_tag25h9_create()
        # self.lib_tag25h7_create()

        # Tag detector
        self.lib.apriltag_detector_create.restype = ctypes.POINTER(_ApriltagDetector)
        detector = self.lib.apriltag_detector_create()
        # detector.contents.nthreads = int(nthreads)
        # detector.contents.quad_decimate = float(quad_decimate)
        # detector.contents.quad_sigma = float(quad_sigma)
        # detector.refine_edges = int(refine_edges)
        # detector.refine_decode = int(refine_decode)
        # detector.refine_pose = int(refine_pose)

        # Add tag family to detector
        self.lib.apriltag_detector_add_family_bits(detector, family, 2)

        # Camera
        camera = Camera()

        while cv2.waitKey(1) != 113:
            img = camera.update()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            assert len(img.shape) == 2
            assert img.dtype == np.uint8

            c_img = convert_image(self.lib, img)
            self.lib.apriltag_detector_detect.restype = ctypes.POINTER(_ZArray)
            results = self.lib.apriltag_detector_detect(detector, c_img)

            for i in range(results.contents.size):
                # Extract the data for each apriltag that was identified
                apriltag = ctypes.POINTER(_ApriltagDetection)()
                self.lib.zarray_get(results, i, ctypes.byref(apriltag))

                tag = apriltag.contents
                homography = _matd_get_array(tag.H).copy()
                center = np.ctypeslib.as_array(tag.c, shape=(2,)).copy()
                corners = np.ctypeslib.as_array(tag.p, shape=(4, 2)).copy()

                # Draw corners
                for corner in corners:
                    pt = (int(corner[0]), int(corner[1]))
                    img = cv2.circle(img, pt, 10, (0, 255, 0), -1)

                # Detection
                # detection = Detection(tag.family.contents.name,
                #                       tag.id,
                #                       tag.hamming,
                #                       tag.goodness,
                #                       tag.decision_margin,
                #                       homography,
                #                       center,
                #                       corners)

            # Clean up
            self.lib.apriltag_detections_destroy(results)
            self.lib.image_u8_destroy(c_img)

            cv2.imshow("Image", img)
