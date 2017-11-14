from os.path import join
from os.path import realpath
import datetime as dt

import numpy as np
import matplotlib.pylab as plt

from prototype.utils.gps import latlon_diff
from prototype.utils.filesystem import walkdir


def load_calib_file(filepath):
    """Load calibration file and parse into a python dictionary

    Parameters
    ----------
    filepath : str
        File path to calibration file

    Returns
    -------
    data : dict
        Dictionary of calibration file

    """
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
    """VO data sequnce"""
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
        """Load time data

        Parameters
        ----------
        sequence_data_path : str
            Path to where VO sequence data is

        """
        time_file = open(join(sequence_data_path, "times.txt"), "r")
        self.time = np.array([float(line) for line in time_file])
        time_file.close()

    def _load_calibration(self, sequence_data_path):
        """Load camera calibration data

        Parameters
        ----------
        sequence_data_path : str
            Path to where VO sequence data is

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
        """Load image files names

        Note: this function only obtains the image file names, it does not load
        images to memory

        Parameters
        ----------
        sequence_data_path : str
            Path to where VO sequence data is

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
        """Load ground truth

        Parameters
        ----------
        sequence : str
            VO sequence data
        dataset_path : str
            Datatset path

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
    """Raw data sequence

    Attributes
    ----------
    base_dir : str
        Base directory
    date : str
        Raw data sequence date
    drive : str
        Raw data drive sequence

    image_00_files : :obj`list` of :obj`str`
        List of image file string for camera 0
    image_01_files : :obj`list` of :obj`str`
        List of image file string for camera 1
    image_02_files : :obj`list` of :obj`str`
        List of image file string for camera 2
    image_03_files : :obj`list` of :obj`str`
        List of image file string for camera 3

    oxts : :obj`list` of :obj`dict`
        Where each element

            data = {
                "lat": data[0], "lon": data[1], "alt": data[2],
                "roll": data[3], "pitch": data[4], "yaw": data[5],
                "vn": data[6], "ve": data[7],
                "vf": data[8], "vl": data[9], "vu": data[10],
                "ax": data[11], "ay": data[12], "az": data[13],
                "af": data[14], "al": data[15], "au": data[16],
                "wx": data[17], "wy": data[18], "wz": data[19],
                "wf": data[20], "wl": data[21], "wu": data[22],
                "pos_accuracy": data[23], "vel_accuracy": data[24],
                "navstat": data[25], "numsats": data[26],
                "posmode": data[27], "velmode": data[28],
                "orimode": data[29]
            }

    calib_camcam : str
        Camera to camera calibration
    calib_imu2velo : str
        IMU to Velodyne calibration
    calib_velo2cam : str
        Velodyne to camera calibration

    timestamps : :obj`list` of :obj`DateTime`
        List of `DateTime` timestamps

    Parameters
    ----------
    base_dir : str
        Base directory
    date : str
        Raw data sequence date
    drive : str
        Raw data drive sequence

    """
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
        """Load image files names

        Note: this function only obtains the image file names, it does not load
        images to memory

        Parameters
        ----------
        sequence_data_path : str
            Path to where VO sequence data is

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
        """Load Oxts data"""
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
                "ax": data[11], "ay": data[12], "az": data[13],
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
        """Load calibration files"""
        self.calib_cam2cam = load_calib_file(join(self.base_dir, self.date,
                                                  "calib_cam_to_cam.txt"))
        self.calib_imu2velo = load_calib_file(join(self.base_dir, self.date,
                                                   "calib_imu_to_velo.txt"))
        self.calib_velo2cam = load_calib_file(join(self.base_dir, self.date,
                                                   "calib_velo_to_cam.txt"))

    def _load_timestamps(self):
        """Load timestamps from file"""
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

    def get_dt(self, index):
        """Get dt"""
        if index > 0 and index < (len(self.timestamps) - 1):
            t_prev = self.timestamps[index - 1]
            t_now = self.timestamps[index]
            dt = (t_now - t_prev).total_seconds()
            return dt
        else:
            return 0.0

    def get_local_pos_true(self):
        """Get true local position"""
        lat_ref = self.oxts[0]['lat']
        lon_ref = self.oxts[0]['lon']
        alt_ref = self.oxts[0]['alt']
        local_pos = np.zeros((3, 1))

        for i in range(1, len(self.oxts)):
            # Calculate position relative to home point
            lat = self.oxts[i]['lat']
            lon = self.oxts[i]['lon']
            height = self.oxts[i]['alt'] - alt_ref
            dist_N, dist_E = latlon_diff(lat_ref, lon_ref, lat, lon)

            pos = np.array([[dist_E], [dist_N], [height]])
            local_pos = np.hstack((local_pos, pos))

        return local_pos

    def get_vel_true(self, index=None):
        """Get true velocity"""
        if index is None:
            vel_x = [data["vf"] for data in self.oxts]
            vel_y = [data["vl"] for data in self.oxts]
            vel_z = [data["vu"] for data in self.oxts]
        else:
            vel_x = [self.oxts[index]["vf"]]
            vel_y = [self.oxts[index]["vl"]]
            vel_z = [self.oxts[index]["vu"]]

        return np.array([vel_x, vel_y, vel_z])

    def get_ang_vel_true(self, index=None):
        """Get true angular velocity"""
        if index is None:
            gyro_x = [data["wf"] for data in self.oxts]
            gyro_y = [data["wl"] for data in self.oxts]
            gyro_z = [data["wu"] for data in self.oxts]
        else:
            gyro_x = [self.oxts[index]["wf"]]
            gyro_y = [self.oxts[index]["wl"]]
            gyro_z = [self.oxts[index]["wu"]]

        return np.array([gyro_x, gyro_y, gyro_z])

    def get_accel_true(self, index=None):
        """Get true acceleration"""
        if index is None:
            accel_x = [data["af"] for data in self.oxts]
            accel_y = [data["al"] for data in self.oxts]
            accel_z = [data["au"] for data in self.oxts]
        else:
            accel_x = [self.oxts[index]["af"]]
            accel_y = [self.oxts[index]["al"]]
            accel_z = [self.oxts[index]["au"]]

        return np.array([accel_x, accel_y, accel_z])

    def get_att_true(self, index=None):
        """Get true attitude"""
        if index is None:
            roll = [data["roll"] for data in self.oxts]
            pitch = [data["pitch"] for data in self.oxts]
            yaw = [data["yaw"] for data in self.oxts]
        else:
            roll = [self.oxts[index]["roll"]]
            pitch = [self.oxts[index]["pitch"]]
            yaw = [self.oxts[index]["yaw"]]

        return np.array([roll, pitch, yaw])

    def plot_accelerometer(self):
        """Plot accelerometer"""
        accel_x = [data["af"] for data in self.oxts]
        accel_y = [data["al"] for data in self.oxts]
        accel_z = [data["au"] for data in self.oxts]

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.plot(self.timestamps, accel_x)
        ax2.plot(self.timestamps, accel_y)
        ax3.plot(self.timestamps, accel_z)

        plt.suptitle("Accelerometer")
        ax1.set_xlabel("Date Time")
        ax1.set_ylabel("ms^-2")
        ax2.set_xlabel("Date Time")
        ax2.set_ylabel("ms^-2")
        ax2.set_xlabel("Date Time")
        ax3.set_ylabel("ms^-2")
        ax1.set_xlim([self.timestamps[0], self.timestamps[-1]])
        ax2.set_xlim([self.timestamps[0], self.timestamps[-1]])
        ax3.set_xlim([self.timestamps[0], self.timestamps[-1]])
        fig.tight_layout()

    def plot_gyroscope(self):
        """Plot gyroscope"""
        gyro_x = [data["wf"] for data in self.oxts]
        gyro_y = [data["wl"] for data in self.oxts]
        gyro_z = [data["wu"] for data in self.oxts]

        fig = plt.figure()
        plt.suptitle("Gyroscope")
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.plot(self.timestamps, gyro_x)
        ax1.set_xlabel("Date Time")
        ax1.set_ylabel("rad s^-1")
        ax1.set_xlim([self.timestamps[0], self.timestamps[-1]])

        ax2.plot(self.timestamps, gyro_y)
        ax2.set_xlabel("Date Time")
        ax2.set_ylabel("rad s^-1")
        ax2.set_xlim([self.timestamps[0], self.timestamps[-1]])

        ax3.plot(self.timestamps, gyro_z)
        ax3.set_xlabel("Date Time")
        ax3.set_ylabel("rad s^-1")
        ax3.set_xlim([self.timestamps[0], self.timestamps[-1]])

        fig.tight_layout()

    def plot_ground_truth(self):
        """Plot ground truth"""
        # Home point
        lat_ref = self.oxts[0]['lat']
        lon_ref = self.oxts[0]['lon']
        alt_ref = self.oxts[0]['alt']

        # Calculate position relative to home point
        ground_truth_x = [0.0]
        ground_truth_y = [0.0]
        ground_truth_z = [0.0]

        for i in range(1, len(self.oxts)):
            lat = self.oxts[i]['lat']
            lon = self.oxts[i]['lon']
            alt = self.oxts[i]['alt']

            dist_N, dist_E = latlon_diff(lat_ref, lon_ref, lat, lon)
            height = alt - alt_ref

            ground_truth_x.append(dist_E)
            ground_truth_y.append(dist_N)
            ground_truth_z.append(height)

        # Plot
        fig = plt.figure()
        plt.suptitle("Ground Truth")
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(ground_truth_x, ground_truth_y)
        ax1.axis('equal')
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")

        ax2.plot(self.timestamps, ground_truth_z)
        ax2.set_xlabel("Date Time")
        ax2.set_ylabel("Height (m)")

        fig.tight_layout()
