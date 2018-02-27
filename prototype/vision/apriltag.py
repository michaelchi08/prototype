#!/usr/bin/env python
import ctypes
from ctypes import POINTER
from ctypes import Structure
from ctypes import c_int
from ctypes import c_int32
from ctypes import c_int64
from ctypes import c_uint8
from ctypes import c_float
from ctypes import c_double

import cv2
import numpy as np


# Import AprilTag C library
LIBPATH = "/usr/local/lib/libapriltag.so"
lib = ctypes.CDLL(LIBPATH)


class _ImageU8(Structure):
    """Wraps image_u8 C struct."""
    _fields_ = [
        ('width', c_int),
        ('height', c_int),
        ('stride', c_int),
        ('buf', POINTER(c_uint8))
    ]


class _Matd(Structure):
    """Wraps matd C struct."""
    _fields_ = [
        ('nrows', c_int),
        ('ncols', c_int),
        ('data', ctypes.c_double*1),
    ]


class _ZArray(Structure):
    """Wraps zarray C struct."""
    _fields_ = [
        ('el_sz', ctypes.c_size_t),
        ('size', c_int),
        ('alloc', c_int),
        ('data', ctypes.c_void_p)
    ]


class _AprilTagFamily(Structure):
    """Wraps apriltag_family C struct."""
    _fields_ = [
        ('ncodes', c_int32),
        ('codes', POINTER(c_int64)),
        ('black_border', c_int32),
        ('d', c_int32),
        ('h', c_int32),
        ('name', ctypes.c_char_p),
    ]


class _AprilTagDetection(Structure):
    """Wraps apriltag_detection C struct."""
    _fields_ = [
        ('family', POINTER(_AprilTagFamily)),
        ('id', c_int),
        ('hamming', c_int),
        ('goodness', c_float),
        ('decision_margin', c_float),
        ('H', POINTER(_Matd)),
        ('c', ctypes.c_double*2),
        ('p', (ctypes.c_double*2)*4)
    ]


class _AprilTagDetector(Structure):
    """Wraps apriltag_detector C struct."""
    _fields_ = [
        ('nthreads', c_int),
        ('quad_decimate', c_float),
        ('quad_sigma', c_float),
        ('refine_edges', c_int),
        ('refine_decode', c_int),
        ('refine_pose', c_int),
        ('debug', c_int),
        ('quad_contours', c_int),
    ]


def ptr_to_array2d(datatype, ptr, rows, cols):
    array_type = (datatype * cols) * rows
    array_buf = array_type.from_address(ctypes.addressof(ptr))
    return np.ctypeslib.as_array(array_buf, shape=(rows, cols))


def image_u8_get_array(img_ptr):
    return ptr_to_array2d(c_uint8,
                          img_ptr.contents.buf.contents,
                          img_ptr.contents.height,
                          img_ptr.contents.stride)


def matd_get_array(mat_ptr):
    return ptr_to_array2d(c_double,
                          mat_ptr.contents.data,
                          int(mat_ptr.contents.nrows),
                          int(mat_ptr.contents.ncols))


def convert_image(img):
    height = img.shape[0]
    width = img.shape[1]

    lib.image_u8_create.restype = ctypes.POINTER(_ImageU8)
    c_img = lib.image_u8_create(width, height)
    tmp = image_u8_get_array(c_img)

    # Copy the opencv image into the destination array, accounting for the
    # difference between stride & width.
    tmp[:, :width] = img

    # tmp goes out of scope here but we don't care because
    # the underlying data is still in c_img.
    return c_img


def tag36h11_create():
    lib.tag36h11_create.restype = POINTER(_AprilTagFamily)
    family = lib.tag36h11_create()
    return family


def tag36h10_create():
    lib.tag36h10_create.restype = POINTER(_AprilTagFamily)
    family = lib.tag36h10_create()
    return family


def tag36artoolkit_create():
    lib.tag36artoolkit_create.restype = POINTER(_AprilTagFamily)
    family = lib.tag36artoolkit_create()
    return family


def tag25h9_create():
    lib.tag25h9_create.restype = POINTER(_AprilTagFamily)
    family = lib.tag25h9_create()
    return family


def tag25h7_create():
    lib.tag25h7_create.restype = POINTER(_AprilTagFamily)
    family = lib.tag25h7_create()
    return family


def apriltag_detector_create():
    lib.apriltag_detector_create.restype = POINTER(_AprilTagDetector)
    detector = lib.apriltag_detector_create()
    return detector


def apriltag_detector_add_family(detector, family):
    lib.apriltag_detector_add_family_bits(detector, family, 2)


def apriltag_detector_detect(detector, c_img):
    lib.apriltag_detector_detect.restype = POINTER(_ZArray)
    results = lib.apriltag_detector_detect(detector, c_img)
    return results


def apriltag_detections_destroy(detections):
    lib.apriltag_detections_destroy(detections)


def image_u8_destroy(c_img):
    lib.image_u8_destroy(c_img)


class AprilTag:
    def __init__(self, **kwargs):
        self.family = kwargs["family"]
        self.id = kwargs["id"]
        self.hamming = kwargs["hamming"]
        self.goodness = kwargs["goodness"]
        self.decision_margin = kwargs["decision_margin"]
        self.H = matd_get_array(kwargs["H"]).copy()
        self.c = np.ctypeslib.as_array(kwargs["c"], shape=(2,)).copy()
        self.p = np.ctypeslib.as_array(kwargs["p"], shape=(4, 2)).copy()

    def draw_corners(self, img):
        for corner in self.p:
            pt = (int(corner[0]), int(corner[1]))
            img = cv2.circle(img, pt, 10, (0, 255, 0), -1)

    def draw_id(self, img):
        # tag_width_px = self.p[0][0] - self.p[1][0]

        text = str(self.id)
        center = (int(self.c[0] - 15), int(self.c[1]))
        font = cv2.FONT_HERSHEY_DUPLEX
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(img, text, center, font, 0.7, color, thickness)


class AprilTagDetector:
    """Custom AprilTag library wrapper"""

    def __init__(self, **kwargs):
        # Tag detector
        self.detector = apriltag_detector_create()
        self.detector.contents.nthreads = kwargs.get("nthreads", 1)
        self.detector.contents.quad_decimate = kwargs.get("quad_decimate", 1.0)
        self.detector.contents.quad_sigma = kwargs.get("quad_sigma", 0.0)
        self.detector.refine_edges = kwargs.get("refine_edges", 1)
        self.detector.refine_decode = kwargs.get("refine_decode", 0)
        self.detector.refine_pose = kwargs.get("refine_pose", 0)

        # Tag family
        family_str = kwargs.get("family", "36h11")
        family = None
        if family_str == "36h11":
            family = tag36h11_create()
        elif family_str == "36h10":
            family = tag36h10_create()
        elif family_str == "36artoolkit":
            family = tag36artoolkit_create()
        elif family_str == "25h9":
            family = tag25h9_create()
        elif family_str == "25h7":
            family = tag25h7_create()
        else:
            raise RuntimeError("Unrecognized tag family: %s", family_str)

        # Add tag family to detector
        apriltag_detector_add_family(self.detector, family)

    def detect(self, img):
        # Make sure image is grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create an ImageU8 instance for the detector and detect tags
        imgu8 = convert_image(gray_img)
        detections = apriltag_detector_detect(self.detector, imgu8)

        # Loop through detections
        results = []
        for i in range(detections.contents.size):
            # Extract the data for each apriltag that was identified
            tag = ctypes.POINTER(_AprilTagDetection)()
            el_sz = detections.contents.el_sz
            data = detections.contents.data
            ctypes.memmove(ctypes.byref(tag), data + i * el_sz, el_sz)

            # Create AprilTag object instance
            results.append(
                AprilTag(family=tag.contents.family,
                         id=tag.contents.id,
                         hamming=tag.contents.hamming,
                         goodness=tag.contents.goodness,
                         decision_margin=tag.contents.decision_margin,
                         H=tag.contents.H,
                         c=tag.contents.c,
                         p=tag.contents.p)
            )

        # Clean up - THIS IS VERY IMPORTANT! ELSE MEMORY LEAKS!
        apriltag_detections_destroy(detections)
        image_u8_destroy(imgu8)

        return results
