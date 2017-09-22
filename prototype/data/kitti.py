from os.path import join

import numpy as np


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

        # Load data
        self._load_time(self.sequence_data_path)
        self._load_calib(self.sequence_data_path)

    def _load_time(self, sequence_data_path):
        """ Load time data """
        time_file = open(join(sequence_data_path, "times.txt"), "r")
        self.time = np.array([float(line) for line in time_file])
        time_file.close()

    def _load_calib(self, sequence_data_path):
        """ Load camera calibration data """
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
