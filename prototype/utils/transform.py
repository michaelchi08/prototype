import numpy as np

from prototype.utils.utils import deg2rad
from prototype.utils.euler import rotz
from prototype.utils.euler import rotx


class TVec3:
    """Transform Vec3"""

    def __init__(self, frame_to, frame_from, x):
        """ Constructor

        Args:

            to (str): Frame to
            from (str): Frame from

        """
        self.frame_to = frame_to
        self.frame_from = frame_from
        self.x = x.reshape((3, 1))

    def data(self):
        """ """
        return self.x


class TPos(TVec3):
    """Transform Position"""

    def __init__(self, frame_to, frame_from, x):
        self.super(frame_to, frame_from, x)


class TVel(TVec3):
    """Transform Velocity"""

    def __init__(self, frame_to, frame_from, x):
        self.super(frame_to, frame_from, x)


class TAngVel(TVec3):
    """Transform Position"""

    def __init__(self, frame_to, frame_from, x):
        self.super(frame_to, frame_from, x)


class Transform:
    """Transform"""

    def __init__(self, frame_to, frame_from, **kwargs):
        """ Constructor

        Args:

            to (str): Frame to
            from (str): Frame from

        kwargs:

            R (np.array - 3x3): Rotation matrix
            t (np.array - 3x1): Translation vector

        """
        self.frame_to = frame_to
        self.frame_from = frame_from

        data = kwargs.get("data", None)
        if data is not None:
            self.R = data[0:3, 0:3].reshape((3, 3))
            self.t = data[0:3, 3].reshape((3, 1))

        else:
            self.R = kwargs.get("R", np.zeros((3, 3)))
            self.t = kwargs.get("t", np.zeros((3, 1)))
            self.R = self.R.reshape((3, 3))
            self.t = self.t.reshape((3, 1))

    def data(self):
        """Obtain transform as a 4x4 homogeneous transform matrix

        Parameters
        ----------

        Returns
        -------

            Homogenous transform matrix (np.array - 4x4)

        """
        return np.block([[self.R, self.t], [0.0, 0.0, 0.0, 1.0]])

    def __mul__(self, x):
        """ Apply transform to x

        Args:

            x (np.array): Matrix or vector to be transformed

        Returns:

            Transformed vector x

        """
        # Transform matrix
        T = self.data()

        # Input is transform class
        if type(x) == Transform:
            return Transform(self.frame_to,
                             x.frame_from,
                             data=np.dot(x.data(), self.data()))

        # Input is rotation matrix
        elif x.shape == (3, 3):
            R = x
            T_x = np.block([[R, np.zeros((3, 1))], [0.0, 0.0, 0.0, 1.0]])
            return np.dot(T, T_x)[0:3, 0:3]

        # Input is vector (not in homogeneous coordinates)
        elif len(x) != 4 and len(x) == 3:
            x = x.ravel()
            x = np.array([[x[0]], [x[1]], [x[2]], [1.0]])
            return np.dot(T, x)[0:3]

        # Input is vector (in homogeneous coordinates)
        elif x.shape == (4, 1):
            return np.dot(T, x)

        # Input is matrix
        elif x.shape == (4, 4):
            return np.dot(T, x)


# Useful default transforms
R_camera_global = np.dot(rotx(deg2rad(-90.0)), rotz(deg2rad(-90.0)))
R_global_camera = np.dot(rotz(deg2rad(90.0)), rotx(deg2rad(90.0)))
T_camera_global = Transform(
    "camera",
    "global",
    R=R_camera_global
)
T_global_camera = Transform(
    "global",
    "camera",
    data=np.linalg.inv(T_camera_global.data())
)
