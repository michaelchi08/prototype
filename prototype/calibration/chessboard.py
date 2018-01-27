from math import pi

import cv2
import numpy as np

from prototype.utils.euler import euler2rot


class Chessboard:
    """ Chessboard

    Attributes
    ----------
    nb_rows : float
        Number of rows
    nb_cols : float
        Number of columns
    R_G : np.array
        Rotation in global frame
    t_G : np.array
        Translation in global frame
    grid_points : np.array
        Matrix of Nx2 grid points

    Parameters
    ----------
    nb_rows : float
        Number of rows
    nb_cols : float
        Number of columns
    R_G : np.array
        Rotation in global frame
    t_G : np.array
        Translation in global frame

    """
    def __init__(self, **kwargs):
        self.nb_rows = kwargs.get("nb_rows", 10)
        self.nb_cols = kwargs.get("nb_cols", 10)
        self.square_size = kwargs.get("square_size", 0.1)

        # Chessboard orientation and position
        self.R_BG = kwargs.get("R_BG", euler2rot([-pi / 2, 0.0, -pi / 2], 321))
        self.t_G = kwargs.get("t_G", np.zeros(3))

        # 2D and 3D Grid_points, Starting from top left corner to bottom right
        self.grid_points2d = self.create_grid_points2d()
        self.grid_points3d = self.create_grid_points3d()

        # Create object points for calibration
        self.object_points = self.create_object_points()

    def create_grid_points2d(self):
        """ Create 2D grid_points """
        grid_points2d = []
        for i in range(self.nb_rows):
            for j in range(self.nb_rows):
                grid_points2d.append([i, j])
        grid_points2d = self.square_size * np.array(grid_points2d)

        return grid_points2d

    def create_grid_points3d(self):
        """ Create 3D grid_points """
        T_BG = np.array([
            [self.R_BG[0, 0], self.R_BG[0, 1], self.R_BG[0, 2], self.t_G[0]],
            [self.R_BG[1, 0], self.R_BG[1, 1], self.R_BG[2, 2], self.t_G[1]],
            [self.R_BG[2, 0], self.R_BG[2, 1], self.R_BG[2, 2], self.t_G[2]],
            [0.0, 0.0, 0.0, 1.0]
        ])

        grid_points3d = []
        for point in self.grid_points2d:
            p = np.array([point[0], point[1], 0.0, 1.0])
            p_G = np.dot(T_BG, p)[0:3]
            grid_points3d.append(p_G)
        grid_points3d = np.array(grid_points3d)

        return grid_points3d

    def create_object_points(self):
        """ Create object points """
        # Hard-coding object points - assuming chessboard is origin by
        # setting chessboard in the x-y plane (where z = 0).
        object_points = []
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                pt = [j * self.square_size, i * self.square_size, 0.0]
                object_points.append(pt)
        object_points = np.array(object_points)
        return object_points

    def draw_corners(self, img):
        """ Draw chessboard corners to image

        Parameters
        ----------
        img : np.array
            Image frame

        Returns
        -------
        Image with chessboard corners drawn

        """
        # Find the chessboard corners
        img = np.array(img) # Make a copy
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (self.nb_cols, self.nb_rows)
        ret, corners = cv2.findChessboardCorners(gray_img, size, None)
        if ret is False:
            return img

        # Draw image
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), crit)
        img = cv2.drawChessboardCorners(img, size, corners, ret)

        return img

    def draw_coord_frame(self, img, corners, K, D=np.zeros(4)):
        """ Draw coordinate frame onto image

        Parameters
        ----------
        img : np.array
            Input image
        corners : np.array
            Chessboard corners
        K : np.array
            Camera matrix K
        D : np.array
            Rad-Tan distortion parameters

        Returns
        -------
        img : np.array
            Image with coordinate frame drawn on

        """
        # Coordinate frame
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
        axis = self.square_size * axis  # Scale chessboard sq size

        # Solve PnP and project coordinate frame to image plane
        T, rvec, tvec = self.solvepnp(corners, K, D)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, D)

        # Draw coordinate frame
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))
        img = np.array(img)  # Make a copy
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)

        return img

    def draw_viz(self, img, corners, K):
        """ Draw visualization

        Parameters
        ----------
        img : np.array
            Input image
        corners : np.array
            Chessboard corners
        K : np.array
            Camera matrix K

        Returns
        -------
        img : np.array
            Output Image

        """
        img = self.draw_corners(img)
        img = self.draw_coord_frame(img, corners, K)
        return img

    def find_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH
        flags += cv2.CALIB_CB_NORMALIZE_IMAGE
        flags += cv2.CALIB_CB_FAST_CHECK
        size = (self.nb_cols, self.nb_rows)

        ret, img_points = cv2.findChessboardCorners(gray, size, None, flags)
        if ret is True:
            return img_points
        else:
            print("Failed to detected chessboard in image!")
            return None

    def solvepnp(self, corners, K, D=np.zeros(4)):
        """ Calculate transform from camera to chessboard

        Parameters
        ----------
        corners : np.array
            Chessboard corners (starting top left, to bottom right)
        K : np.array
            Camera matrix K
        D : np.array
            Rad-tan distortion coefficients

        Returns
        -------
        T, rev, tvec : np.array, np.array, np.array
            Transform from camera to chessboard, and the corresponding
            Rodrigues rotation vector and translation vector

        """
        # Calculate transformation matrix
        retval, rvec, tvec = cv2.solvePnP(self.object_points,
                                          corners,
                                          K,
                                          D,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if retval is False:
            raise RuntimeError("solvePnP failed!")

        # Convert rotation vector to matrix
        R = np.zeros((3, 3))
        cv2.Rodrigues(rvec, R)

        # Form transformation matrix
        T = np.array([[R[0][0], R[0][1], R[0][2], tvec[0]],
                      [R[1][0], R[1][1], R[1][2], tvec[1]],
                      [R[2][0], R[2][1], R[2][2], tvec[2]],
                      [0.0, 0.0, 0.0, 1.0]])

        return T, rvec, tvec

    def calc_corner_positions(self, corners, K, D=np.zeros(4)):
        """ Calculate chessboard corner position

        Having calculated the passive transform from camera to chessboard via
        SolvePnP, this function takes the transform and calculates the 3D
        position of each chessboard corner.

        Returns
        -------
        X : np.array (Nx3)
            N Chessboard corners in 3D as a matrix

        """
        T_o_t, _, _ = self.solvepnp(corners, K, D=np.zeros(4))

        nb_obj_pts = self.object_points.shape[0]
        ones = np.ones((nb_obj_pts, 1))
        obj_pts_homo = np.block([self.object_points, ones])
        obj_pts_homo = obj_pts_homo.T

        X = np.dot(T_o_t, obj_pts_homo)
        X = X.T
        X = X[:, 0:3]

        return X
