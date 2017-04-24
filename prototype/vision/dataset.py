#!/usr/bin/env python3
import numpy as np
from numpy import dot

from prototype.utils.math import nwu2edn
from prototype.utils.math import euler2rot
from prototype.vision.utils import projection_matrix


class TestCamera(object):
    def __init__(self):
        self.dt = 0.0
        self.hz = 10
        self.frame = 0

    def update(self, dt):
        self.dt += dt

        if self.dt > (1.0 / self.hz):
            self.dt = 0.0
            self.frame += 1
            return True

        return False

    def check_features(self, dt, features, rpy, t):
        observed = []

        # pre-check
        if self.update(dt) == False:
            return True

        # rotation matrix - convert from nwu to edn then to rotation matrix R
        rpy_edn = nwu2edn(rpy)
        R = euler2rot(rpy_edn, 123)

        # translation - convert translation from nwu to edn
        t_edn = nwu2edn(t)

        # projection matrix
        P = projection_matrix(self.K, R, dot(-R, t_edn))

        # check which features in 3d are observable from camera
        for i in range(features.shape[1]):
            # convert feature in NWU to EDN coordinate system
            f_3d = features[0:4, i]
            f_3d_edn[0] = -f_3d[1]
            f_3d_edn[1] = -f_3d[2]
            f_3d_edn[2] = f_3d[0]
            f_3d_edn[3] = 1.0

            # project 3D world point to 2D image plane
            f_2d = dot(P, f_3d_edn)

            # check to see if feature is valid and infront of camera
            if f_2d[2] >= 1.0:
                # normalize pixels
                f_2d[0] = f_2d[0] / f_2d[2]
                f_2d[1] = f_2d[1] / f_2d[2]
                f_2d[2] = f_2d[2] / f_2d[2]

                # check to see if feature observed is within image plane
                if (f_2d[0] < self.image_width) and (f_2d[0] > 0.0):
                    if (f_2d[1] < self.image_height) and (f_2d[1] > 0):
                    observed.append((f_2d[0:2], f_3d[0:3]))

    def prep_header(self, output_file):
        header = ["time_step", "x", "y", "theta"]
        output_file.write(header.join(",") + "\n")

    def record_observation(self, output_file, x):
        output_file.write(x.join(",") + "\n")

    def calculate_circle_angular_velocity(self, r, v):
        dist = 2 * pi * r
        time = dist / v
        return (2 * M_PI) / time

    def generateRandom3DFeatures(self, nb_features, x_bounds, y_bounds, z_bounds):
        features = []

        # generate random 3d features
        for i in range(nb_features):
            point = [0, 0, 0, 0]
            point[0] = randf(x_bounds[0], x_bounds[1])
            point[1] = randf(y_bounds[0], y_bounds[1])
            point[2] = randf(z_bounds[0], z_bounds[1])
            point[3] = 1.0
            features.append(point)

    # def record3DFeatures(self, output_path, features):
#   return mat2csv(output_path,
#                  features.block(0, 0, 3, features.cols()).transpose())
# }

    # def recordObservedFeatures(time, x, outputpath, observed):
#   // open file
#   if (outfile.good() != true) {
#     log_err("Failed to open file [%s] to record observed features!",
#             output_path.c_str())
#     return -1
#   }
#
#   // time and number of observed features
#   outfile << time << std::endl
#   outfile << observed.size() << std::endl
#   outfile << x(0) << "," << x(1) << "," << x(2) << std::endl
#
#   // features
#   for (auto feature : observed) {
#     // feature in image frame
#     f_2d = feature.first.transpose()
#     outfile << f_2d(0) << "," << f_2d(1) << std::endl
#
#     // feature in world frame
#     f_3d = feature.second.transpose()
#     outfile << f_3d(0) << "," << f_3d(1) << "," << f_3d(2) << std::endl
#   }
#
#   // clean up
#   outfile.close()
#
#   return 0
# }

# def generateTestData(self, save_path):

#   // mkdir calibration directory
#   retval = mkdir(save_path.c_str(), ACCESSPERMS)
#   if (retval != 0) {
#     switch (errno) {
#       case EACCES: log_err(MKDIR_PERMISSION_DENIED, save_path.c_str()) break
#       case ENOTDIR: log_err(MKDIR_INVALID, save_path.c_str()) break
#       case EEXIST: log_err(MKDIR_EXISTS, save_path.c_str()) break
#       default: log_err(MKDIR_FAILED, save_path.c_str()) break
#     }
#     return -2
#   }

#   // setup
#   output_file.open(save_path + "/state.dat")
#   prep_header(output_file)
#   index_file.open(save_path + "/index.dat")
#   calculate_circle_angular_velocity(0.5, 1.0, w)
#   this->generateRandom3DFeatures(features)
#   this->record3DFeatures("/tmp/test/features.dat", features)
#
#   // initialize states
#   dt = 0.01
#   time = 0.0
#   x << 0.0, 0.0, 0.0
#   u << 1.0, w
#
#   for (int i = 0 i < 300 i++) {
#     // update state
#     x = two_wheel_model(x, u, dt)
#     time += dt
#
#     // check features
#     rpy << 0.0, 0.0, x(2)
#     t << x(0), x(1), 0.0
#     if (this->camera.checkFeatures(dt, features, rpy, t, observed) == 0) {
#       oss.str("")
#       oss << "/tmp/test/observed_" << this->camera.frame << ".dat"
#       this->recordObservedFeatures(time, x, oss.str(), observed)
#
#       index_file << oss.str() << std::endl
#     }
#
#     // record state
#     record_observation(output_file, x)
#   }
#
#   // clean up
#   output_file.close()
#   index_file.close()
#   return 0
# }
