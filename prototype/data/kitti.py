from os.path import join
from os.path import realpath

import numpy as np

from prototype.utils.filesystem import walkdir


class VOSequence:
    def __init__(self, dataset_path, sequence):
        # Dataset path and information
        self.dataset_path = dataset_path
        self.sequence = sequence
        self.sequence_data_path = join(self.dataset_path, "sequences", sequence)

        # Time
        self.time = np.array([])

        # Camera Projection Matrix (3, 4)
        self.P0 = np.array([])
        self.P1 = np.array([])
        self.P2 = np.array([])
        self.P3 = np.array([])
        self.Tr = np.array([])

        # Images
        self.image_0_files = []
        self.image_1_files = []

        # Ground Truth
        self.ground_truth = np.array([])

        # Load data
        self._load_time(self.sequence_data_path)
        self._load_calibration(self.sequence_data_path)
        self._load_image_file_names(self.sequence_data_path)
        self._load_ground_truth(self.dataset_path, self.sequence)

    def _load_time(self, sequence_data_path):
        """ Load time data

        Args:

            sequence_data_path (str): Path to where VO sequence data is

        """
        time_file = open(join(sequence_data_path, "times.txt"), "r")
        self.time = np.array([float(line) for line in time_file])
        time_file.close()

    def _load_calibration(self, sequence_data_path):
        """ Load camera calibration data

        Args:

            sequence_data_path (str): Path to where VO sequence data is

        """
        calib_file = open(join(sequence_data_path, "calib.txt"), "r")

        # Parse calibration file
        for line in calib_file:
            elements = line.strip().split(" ")
            token = elements[0]
            data = np.array([float(el) for el in elements[1:]])
            data = data.reshape((3, 4))

            if token == "P0:":
                self.P0 = data
            elif token == "P1:":
                self.P1 = data
            elif token == "P2:":
                self.P2 = data
            elif token == "P3:":
                self.P3 = data
            elif token == "Tr:":
                self.Tr = data

        calib_file.close()

    def _load_image_file_names(self, sequence_data_path):
        """ Load image files names

        Note: this function only obtains the image file names, it does not load
        images to memory

        Args:

            sequence_data_path (str): Path to where VO sequence data is

        """
        image_0_path = join(sequence_data_path, "image_0")
        image_1_path = join(sequence_data_path, "image_1")

        self.image_0_files = walkdir(image_0_path, ".png")
        self.image_1_files = walkdir(image_1_path, ".png")

        self.image_0_files.sort(
            key=lambda f: int("".join(filter(str.isdigit, f)))
        )
        self.image_1_files.sort(
            key=lambda f: int("".join(filter(str.isdigit, f)))
        )

    def _load_ground_truth(self, dataset_path, sequence):
        # Build ground truth file path
        ground_truth_dir = realpath(join(dataset_path, sequence, "../poses"))
        ground_truth_fname = join(ground_truth_dir, sequence + ".txt")
        ground_truth_file = open(ground_truth_fname, "r")

        # Parse ground truth file
        ground_truth = []
        ground_truth_lines = ground_truth_file.readlines()

        for line in ground_truth_lines:
            line = line.strip().split()
            x = float(line[3])
            y = float(line[7])
            z = float(line[11])
            ground_truth.append([x, y, z])

        # Finish up
        ground_truth_file.close()
        self.ground_truth = np.array(ground_truth)
