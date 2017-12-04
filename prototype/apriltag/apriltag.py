#!/usr/bin/env python
"""Python wrapper for C version of apriltags. This program creates two
classes that are used to detect apriltags and extract information from
them. Using this module, you can identify all apriltags visible in an
image, and get information about the location and orientation of the
tags.

Original author: Isaac Dulin, Spring 2016
Updates: Matt Zucker, Fall 2016

"""
import ctypes
import collections
import os
import re

import numpy
import cv2

from prototype.apriltag.apriltag_structs import _ImageU8
from prototype.apriltag.apriltag_structs import _Matd
from prototype.apriltag.apriltag_structs import _ZArray
from prototype.apriltag.apriltag_structs import _ApriltagFamily
from prototype.apriltag.apriltag_structs import _ApriltagDetection
from prototype.apriltag.apriltag_structs import _ApriltagDetector


def _ptr_to_array2d(datatype, ptr, rows, cols):
    array_type = (datatype*cols)*rows
    array_buf = array_type.from_address(ctypes.addressof(ptr))
    return numpy.ctypeslib.as_array(array_buf, shape=(rows, cols))


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


DetectionBase = collections.namedtuple(
    'DetectionBase',
    'tag_family'
    'tag_id'
    'hamming'
    'goodness'
    'decision_margin'
    'homography'
    'center'
    'corners'
)


class Detection(DetectionBase):
    """Pythonic wrapper for apriltag_detection which derives from named
    tuple class."""

    _print_fields = [
        'Family', 'ID', 'Hamming error', 'Goodness',
        'Decision margin', 'Homography', 'Center', 'Corners'
    ]

    _max_len = max(len(field) for field in _print_fields)

    def tostring(self, indent=0):
        """Converts this object to a string with the given level of
        indentation."""
        rval = []
        indent_str = ' '*(self._max_len+2+indent)

        for i, label in enumerate(self._print_fields):

            value = str(self[i])

            if value.find('\n') > 0:
                value = value.split('\n')
                value = [value[0]] + [indent_str+v for v in value[1:]]
                value = '\n'.join(value)

            rval.append('{:>{}s}: {}'.format(
                label, self._max_len+indent, value))

        return '\n'.join(rval)

    def __str__(self):
        return self.tostring()


class DetectorOptions:
    """Convience wrapper for object to pass into Detector
    initializer. You can also pass in the output of an
    argparse.ArgumentParser on which you have called add_arguments."""

    def __init__(self,
                 families='tag36h11',
                 border=1,
                 nthreads=4,
                 quad_decimate=1.0,
                 quad_blur=0.0,
                 refine_edges=True,
                 refine_decode=False,
                 refine_pose=False,
                 debug=False,
                 quad_contours=True):

        self.families = families
        self.border = int(border)

        self.nthreads = int(nthreads)
        self.quad_decimate = float(quad_decimate)
        self.quad_sigma = float(quad_blur)
        self.refine_edges = int(refine_edges)
        self.refine_decode = int(refine_decode)
        self.refine_pose = int(refine_pose)
        self.debug = int(debug)
        self.quad_contours = quad_contours


class Detector:
    """Pythonic wrapper for apriltag_detector. Initialize by passing in the
    output of an argparse.ArgumentParser on which you have called
    add_arguments; or an instance of the DetectorOptions class."""

    def __init__(self, options=None):
        if options is None:
            options = DetectorOptions()
        self.options = options

        # Detect OS to get extension for DLL
        uname0 = os.uname()[0]
        if uname0 == 'Darwin':
            extension = '.dylib'
        else:
            extension = '.so'

        filename = 'libapriltag' + extension

        # Load the C library and store it as a class variable
        # Note: prefer OS install to local!
        try:
            self.libc = ctypes.CDLL(filename)
        except OSError:
            selfdir = os.path.dirname(__file__)
            relpath = os.path.join(selfdir, '../build/lib/', filename)
            if not os.path.exists(relpath):
                raise
            self.libc = ctypes.CDLL(relpath)

        # declare return types of libc function
        self._declare_return_types()

        # create the c-_apriltag_detector object
        self.tag_detector = self.libc.apriltag_detector_create()
        self.tag_detector.contents.nthreads = int(options.nthreads)
        self.tag_detector.contents.quad_decimate = float(options.quad_decimate)
        self.tag_detector.contents.quad_sigma = float(options.quad_sigma)
        self.tag_detector.refine_edges = int(options.refine_edges)
        self.tag_detector.refine_decode = int(options.refine_decode)
        self.tag_detector.refine_pose = int(options.refine_pose)
        if options.quad_contours:
            self.libc.apriltag_detector_enable_quad_contours(
                self.tag_detector,
                1
            )

        self.families = []
        flist = self.libc.apriltag_family_list()

        for i in range(flist.contents.size):
            ptr = ctypes.c_char_p()
            self.libc.zarray_get(flist, i, ctypes.byref(ptr))
            self.families.append(ctypes.string_at(ptr))

        if options.families == 'all':
            families_list = self.families
        elif isinstance(options.families, list):
            families_list = options.families
        else:
            families_list = [n for n in re.split(r'\W+', options.families) if n]

        # add tags
        for family in families_list:
            self.add_tag_family(family)

    def detect(self, img, return_image=False):
        """Run detectons on the provided image. The image must be a grayscale
        image of type numpy.uint8."""
        assert len(img.shape) == 2
        assert img.dtype == numpy.uint8

        c_img = self._convert_image(img)

        # detect apriltags in the image
        detections = self.libc.apriltag_detector_detect(self.tag_detector, c_img)

        # create a pytags_info object
        return_info = []

        for i in range(0, detections.contents.size):

            #extract the data for each apriltag that was identified
            apriltag = ctypes.POINTER(_ApriltagDetection)()
            self.libc.zarray_get(detections, i, ctypes.byref(apriltag))

            tag = apriltag.contents

            homography = _matd_get_array(tag.H).copy()
            center = numpy.ctypeslib.as_array(tag.c, shape=(2,)).copy()
            corners = numpy.ctypeslib.as_array(tag.p, shape=(4, 2)).copy()

            detection = Detection(
                tag.family.contents.name,
                tag.id,
                tag.hamming,
                tag.goodness,
                tag.decision_margin,
                homography,
                center,
                corners)

            #Append this dict to the tag data array
            return_info.append(detection)

        if return_image:

            dimg = self._vis_detections(img.shape, detections)
            return return_info, dimg

        else:

            return return_info


    def add_tag_family(self, name):
        """Add a single tag family to this detector."""
        family = self.libc.apriltag_family_create(name)

        if family:
            family.contents.border = self.options.border
            self.libc.apriltag_detector_add_family(self.tag_detector, family)
        else:
            print('Unrecognized tag family name. Try e.g. tag36h11')

    def _vis_detections(self, shape, detections):
        height, width = shape
        c_dimg = self.libc.image_u8_create(width, height)
        self.libc.apriltag_vis_detections(detections, c_dimg)
        tmp = _image_u8_get_array(c_dimg)

        return tmp[:, :width].copy()

    def _declare_return_types(self):
        self.libc.apriltag_detector_create.restype = ctypes.POINTER(_ApriltagDetector)
        self.libc.apriltag_family_create.restype = ctypes.POINTER(_ApriltagFamily)
        self.libc.apriltag_detector_detect.restype = ctypes.POINTER(_ZArray)
        self.libc.image_u8_create.restype = ctypes.POINTER(_ImageU8)
        self.libc.image_u8_write_pnm.restype = ctypes.c_int
        self.libc.apriltag_family_list.restype = ctypes.POINTER(_ZArray)
        self.libc.apriltag_vis_detections.restype = None

    def _convert_image(self, img):
        height = img.shape[0]
        width = img.shape[1]
        c_img = self.libc.image_u8_create(width, height)

        tmp = _image_u8_get_array(c_img)

        # copy the opencv image into the destination array, accounting for the
        # difference between stride & width.
        tmp[:, :width] = img

        # tmp goes out of scope here but we don't care because
        # the underlying data is still in c_img.
        return c_img


def add_arguments(parser):
    """Add arguments to the given argparse.ArgumentParser object to enable
    passing in the resulting parsed arguments into the initializer for
    Detector. """
    defaults = DetectorOptions()
    show_default = ' (default %(default)s)'

    parser.add_argument('-f', metavar='FAMILIES',
                        dest='families', default=defaults.families,
                        help='Tag families' + show_default)

    parser.add_argument('-B', metavar='N',
                        dest='border', type=int, default=defaults.border,
                        help='Tag border size in pixels' + show_default)

    parser.add_argument('-t', metavar='N',
                        dest='nthreads', type=int, default=defaults.nthreads,
                        help='Number of threads' + show_default)

    parser.add_argument('-x', metavar='SCALE',
                        dest='quad_decimate', type=float,
                        default=defaults.quad_decimate,
                        help='Quad decimation factor' + show_default)

    parser.add_argument('-b', metavar='SIGMA',
                        dest='quad_sigma', type=float,
                        default=defaults.quad_sigma,
                        help='Apply low-pass blur to input' + show_default)

    parser.add_argument('-0', dest='refine_edges', default=True,
                        action='store_false',
                        help='Spend less time trying to align edges of tags')

    parser.add_argument('-1', dest='refine_decode', default=False,
                        action='store_true',
                        help='Spend more time trying to decode tags')

    parser.add_argument('-2', dest='refine_pose', default=False,
                        action='store_true',
                        help='Spend more time trying to localizing tags')

    parser.add_argument('-c', dest='quad_contours', default=False,
                        action='store_true',
                        help='Use new contour-based quad detection')


def main():
    """Test function for this Python wrapper."""
    from argparse import ArgumentParser

    # for some reason pylint complains about members being undefined :(
    # pylint: disable=E1101

    parser = ArgumentParser()
    parser.add_argument('filenames', metavar='IMAGE', nargs='+',
                        help='files to convert')

    add_arguments(parser)

    options = parser.parse_args()
    det = Detector(options)

    for filename in options.filenames:
        orig = cv2.imread(filename)
        if len(orig.shape) == 3:
            gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        else:
            gray = orig

        detections, dimg = det.detect(gray, return_image=True)
        num_detections = len(detections)
        print('Detected {} tags.\n'.format(num_detections))

        for i, detection in enumerate(detections):
            print("Detection {} of {}:".format(i + 1, num_detections))
            print()
            print(detection.tostring(indent=2))
            print()

        if len(orig.shape) == 3:
            overlay = orig / 2 + dimg[:, :, None] / 2
        else:
            overlay = gray / 2 + dimg / 2

        cv2.imshow("win", overlay)
        while cv2.waitKey(5) < 0:
            pass


if __name__ == '__main__':
    main()
