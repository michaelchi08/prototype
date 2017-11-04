import os
from os.path import join
from os.path import realpath
import datetime as dt

import numpy as np

from prototype.utils.filesystem import walkdir


def load_calib_file(filepath):
    """ Load calibration file and parse into a python dictionary """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


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
        """ Load ground truth

        Args:

            sequence (str): VO sequence data

        """
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


class RawSequence:
    def __init__(self, base_dir, date, drive):
        # Dataset path and information
        self.base_dir = base_dir
        self.date = date
        self.drive = drive
        self.drive_dir = date + "_drive_" + drive + "_sync"

        # Image files
        self.image_00_files = []
        self.image_01_files = []
        self.image_02_files = []
        self.image_03_files = []

        # Oxts (GPS INS data)
        self.oxts = []

        # Calibration files
        self.calib_cam2cam = None
        self.calib_imu2velo = None
        self.calib_velo2cam = None

        # Timestamp
        self.timestamps = []

        # Load
        self._load_image_file_names()
        self._load_oxts_data()
        self._load_calibration_files()
        self._load_timestamps()

    def _load_image_file_names(self):
        """ Load image files names

        Note: this function only obtains the image file names, it does not load
        images to memory

        Args:

            sequence_data_path (str): Path to where VO sequence data is

        """
        data_path = join(self.base_dir, self.date, self.drive_dir)

        self.image_00_files = walkdir(join(data_path, "image_00"), ".png")
        self.image_00_files.sort(key=lambda f:
                                 int("".join(filter(str.isdigit, f))))

        self.image_01_files = walkdir(join(data_path, "image_01"), ".png")
        self.image_01_files.sort(key=lambda f:
                                 int("".join(filter(str.isdigit, f))))

        self.image_02_files = walkdir(join(data_path, "image_02"), ".png")
        self.image_02_files.sort(key=lambda f:
                                 int("".join(filter(str.isdigit, f))))

        self.image_03_files = walkdir(join(data_path, "image_03"), ".png")
        self.image_03_files.sort(key=lambda f:
                                 int("".join(filter(str.isdigit, f))))

    def _load_oxts_data(self):
        """ Load Oxts data """
        path = join(self.base_dir, self.date, self.drive_dir, "oxts", "data")
        oxts_files = walkdir(path, ".txt")
        oxts_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        # Load oxts data
        for i in range(len(oxts_files)):
            # Open and load text file
            oxts_file = open(oxts_files[i], "r")
            data = oxts_file.readlines()
            oxts_file.close()

            # Split Oxts data
            data = " ".join(data)
            data = data.split(" ")
            data = [float(x) for x in data]

            data = {
                "lat": data[0], "lon": data[1], "alt": data[2],
                "roll": data[3], "pitch": data[4], "yaw": data[5],
                "vn": data[6], "ve": data[7],
                "vf": data[8], "vl": data[9], "vu": data[10],
                "ax": data[11], "ay": data[12], "ay": data[13],
                "af": data[14], "al": data[15], "au": data[16],
                "wx": data[17], "wy": data[18], "wz": data[19],
                "wf": data[20], "wl": data[21], "wu": data[22],
                "pos_accuracy": data[23], "vel_accuracy": data[24],
                "navstat": data[25], "numsats": data[26],
                "posmode": data[27], "velmode": data[28],
                "orimode": data[29]
            }
            self.oxts.append(data)

    def _load_calibration_files(self):
        """ Load calibration files """
        self.calib_cam2cam = load_calib_file(join(self.base_dir, self.date,
                                                  "calib_cam_to_cam.txt"))
        self.calib_imu2velo = load_calib_file(join(self.base_dir, self.date,
                                                   "calib_imu_to_velo.txt"))
        self.calib_velo2cam = load_calib_file(join(self.base_dir, self.date,
                                                   "calib_velo_to_cam.txt"))

    def _load_timestamps(self):
        """ Load timestamps from file """
        timestamp_file = join(self.base_dir,
                              self.date,
                              self.drive_dir,
                              "oxts",
                              "timestamps.txt")

        # Read and parse the timestamps
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps.append(t)
