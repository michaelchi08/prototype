import ctypes


class _ImageU8(ctypes.Structure):
    """Wraps image_u8 C struct."""
    _fields_ = [
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('stride', ctypes.c_int),
        ('buf', ctypes.POINTER(ctypes.c_uint8))
    ]


class _Matd(ctypes.Structure):
    """Wraps matd C struct."""
    _fields_ = [
        ('nrows', ctypes.c_int),
        ('ncols', ctypes.c_int),
        ('data', ctypes.c_double*1),
    ]


class _ZArray(ctypes.Structure):
    """Wraps zarray C struct."""
    _fields_ = [
        ('el_sz', ctypes.c_size_t),
        ('size', ctypes.c_int),
        ('alloc', ctypes.c_int),
        ('data', ctypes.c_void_p)
    ]


class _ApriltagFamily(ctypes.Structure):
    """Wraps apriltag_family C struct."""
    _fields_ = [
        ('ncodes', ctypes.c_int32),
        ('codes', ctypes.POINTER(ctypes.c_int64)),
        ('black_border', ctypes.c_int32),
        ('d', ctypes.c_int32),
        ('h', ctypes.c_int32),
        ('name', ctypes.c_char_p),
    ]


class _ApriltagDetection(ctypes.Structure):
    """Wraps apriltag_detection C struct."""
    _fields_ = [
        ('family', ctypes.POINTER(_ApriltagFamily)),
        ('id', ctypes.c_int),
        ('hamming', ctypes.c_int),
        ('goodness', ctypes.c_float),
        ('decision_margin', ctypes.c_float),
        ('H', ctypes.POINTER(_Matd)),
        ('c', ctypes.c_double*2),
        ('p', (ctypes.c_double*2)*4)
    ]


class _ApriltagDetector(ctypes.Structure):
    """Wraps apriltag_detector C struct."""
    _fields_ = [
        ('nthreads', ctypes.c_int),
        ('quad_decimate', ctypes.c_float),
        ('quad_sigma', ctypes.c_float),
        ('refine_edges', ctypes.c_int),
        ('refine_decode', ctypes.c_int),
        ('refine_pose', ctypes.c_int),
        ('debug', ctypes.c_int),
        ('quad_contours', ctypes.c_int),
    ]
