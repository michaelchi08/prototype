import os

import cv2
import numpy as np

from prototype.utils.filesystem import walkdir


class PreprocessData:
    """ Preprocess calibration data

    Attributes
    ----------
    image_dir : string
        Image base directory
    images : np.array
        Calibration images
    images_ud : np.array
        Undistorted calibration images

    chessboard : Chessboard
        Chessboard

    intrinsics : CameraIntrinsics
        Camera intrinsics
    corners2d : np.array
        Image corners
    corners3d : np.array
        Image point location
    corners2d_ud : np.array
        Undistorted image corners
    corners3d_ud : np.array
        Undistorted image point location

    """
    def __init__(self, data_type, **kwargs):
        self.data_type = data_type
        if self.data_type == "IMAGES":
            self.images_dir = kwargs["images_dir"]
            self.images = []
            self.images_ud = []
            self.chessboard = kwargs["chessboard"]
            self.intrinsics = kwargs["intrinsics"]
        elif self.data_type == "PREPROCESSED":
            self.data_path = kwargs["data_path"]

        # Result
        self.target_points = []
        self.corners2d = []
        self.corners3d = []
        self.corners2d_ud = []
        self.corners3d_ud = []

    def ideal2pixel(self, points, K):
        """ Ideal points to pixel coordinates

        Parameters
        ----------
        cam_id : int
            Camera ID
        points : np.array
            Points in ideal coordinates

        Returns
        -------
        pixels : np.array
            Points in pixel coordinates

        """
        # Get camera intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Convert ideal points to pixel coordinates
        pixels = []
        nb_points = len(points)
        for p in points.reshape((nb_points, 2)):
            px = (p[0] * fx) + cx
            py = (p[1] * fy) + cy
            pixels.append([px, py])

        return np.array(pixels)

    def get_viz(self, i):
        """ Return a visualization of the original and undistorted image with
        detected chessboard corners and a 3D coordinate axis drawn on the
        images.  The original and undistorted image with the visualizations
        will be stacked horizontally.

        Parameters
        ----------
        i : int
            i-th Image frame

        Returns
        -------
        image_viz : np.array
            Image visualization

        """
        # Visualize original image
        image = self.images[i]
        corners2d = self.corners2d[i]
        K = self.intrinsics.K()
        image = self.chessboard.draw_viz(image, corners2d, K)

        # Visualize undistorted image
        image_ud = self.images_ud[i]
        corners2d_ud = self.corners2d_ud[i]
        K_new = self.intrinsics.K_new
        image_ud = self.chessboard.draw_viz(image_ud, corners2d_ud, K_new)

        # Create visualization
        image_viz = np.hstack((image, image_ud))
        return image_viz

    def preprocess(self):
        """ Preprocess images """
        image_files = walkdir(self.images_dir)
        nb_images = len(image_files)

        # Loop through calibration images
        for i in range(nb_images):
            # Load images and find chessboard corners
            image = cv2.imread(image_files[i])
            corners = self.chessboard.find_corners(image)
            self.images.append(image)

            # Calculate camera to chessboard transform
            K = self.intrinsics.K()
            P_c = self.chessboard.calc_corner_positions(corners, K)
            nb_corners = corners.shape[0]
            self.corners2d.append(corners.reshape((nb_corners, 2)))
            self.corners3d.append(P_c)

            # Undistort corners in camera 0
            corners_ud = self.intrinsics.undistort_points(corners)
            image_ud, K_new = self.intrinsics.undistort_image(image)
            pixels_ud = self.ideal2pixel(corners_ud, K_new)
            self.images_ud.append(image_ud)

            # Calculate camera to chessboard transform
            K_new = self.intrinsics.K_new
            P_c = self.chessboard.calc_corner_positions(pixels_ud, K_new)
            self.corners2d_ud.append(pixels_ud)
            self.corners3d_ud.append(P_c)

        self.corners2d = np.array(self.corners2d)
        self.corners3d = np.array(self.corners3d)
        self.corners2d_ud = np.array(self.corners2d_ud)
        self.corners3d_ud = np.array(self.corners3d_ud)

    def parse_gridpoints_line(self, line, data):
        # Parse line
        elements = line.strip().split(" ")
        elements = [float(x) for x in elements]
        x, y, z = elements[0:3]
        u, v = elements[3:5]

        # Form point 3d and 2d
        point3d = [x, y, z]
        point2d = [u, v]

        # Add to storage
        data["target_points"].append(point3d)
        data["corners3d"].append(point3d)
        data["corners2d"].append(point2d)

    def parse_transform(self, line, data):
        # Parse transform
        elements = line.strip().split(" ")
        elements = [float(x) for x in elements]
        data["T_c_t"] += elements

    def parse_gimbal_angles(self, line, data):
        # Parse gimbal angles
        elements = line.strip().split(" ")
        data["gimbal_angles"] += [float(x) for x in elements]

    def transform_corners(self, data):
        data["T_c_t"] = np.array(data["T_c_t"]).reshape((4, 4))
        data["corners3d"] = np.array(data["corners3d"])
        data["corners2d"] = np.array(data["corners2d"])

        # Transform the 3d points
        # -- Convert 3d points to homogeneous coordinates
        nb_corners = data["corners3d"].shape[0]
        ones = np.ones((nb_corners, 1))
        corners_homo = np.block([data["corners3d"], ones])
        corners_homo = corners_homo.T
        # -- Transform 3d points
        X = np.dot(data["T_c_t"], corners_homo)
        X = X.T
        data["corners3d"] = X[:, 0:3]

    def load_preprocessed_file(self, filepath):
        # Setup
        datafile = open(filepath, "r")
        mode = None

        # Data
        data = {
            "target_points": [],
            "corners3d": [],
            "corners2d": [],
            "gimbal_angles": [],
            "T_c_t": []  # Transform, target to camera
        }

        # Parse file
        for line in datafile:
            line = line.strip()

            if line == "gridpoints:":
                mode = "gridpoints"
            elif line == "tmatrix:":
                mode = "tmatrix"
            elif line == "gimbalangles:":
                mode = "gimbalangles"
            elif line == "end:":
                mode = None
            else:
                if mode == "gridpoints":
                    self.parse_gridpoints_line(line, data)
                elif mode == "tmatrix":
                    self.parse_transform(line, data)
                elif mode == "gimbalangles":
                    self.parse_gimbal_angles(line, data)

        # Finish up
        self.transform_corners(data)
        data["target_points"] = np.array(data["target_points"])
        data["corners2d_ud"] = data["corners2d"]
        data["corners3d_ud"] = data["corners3d"]
        datafile.close()

        return data

    def load_preprocessed(self):
        files = walkdir(self.data_path)
        files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
        if len(files) == 0:
            err_msg = "No data files found in [%s]!" % (self.data_path)
            raise RuntimeError(err_msg)

        for f in files:
            data = self.load_preprocessed_file(f)
            self.target_points.append(data["target_points"])
            self.corners2d.append(data["corners2d"])
            self.corners3d.append(data["corners3d"])

        self.target_points = np.array(self.target_points)
        self.corners2d = np.array(self.corners2d)
        self.corners3d = np.array(self.corners3d)
        self.corners2d_ud = self.corners2d
        self.corners3d_ud = self.corners3d

    def load(self):
        if self.data_type == "IMAGES":
            self.preprocess()
        elif self.data_type == "PREPROCESSED":
            self.load_preprocessed()
