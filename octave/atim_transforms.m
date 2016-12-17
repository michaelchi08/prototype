#!/usr/bin/octave -qf
addpath("./octave/lib/transform")
addpath("./octave/lib/util")


# target position in camera frame (NED)
target = [1.0; 0.0; 0.0]

# camera facing downwards (i.e. pitch -90 deg in NED)
camera_mount = [0.0; deg2rad(-90.0); 0.0];

# camera attitude in inertial frame
camera_attitude = [0.0; deg2rad(10.0); 0.0];  # pitch 10deg, target should be ahead
% camera_attitude = [0.0; deg2rad(-10.0); 0.0];  # pitch 10deg, target should be behind
% camera_attitude = [deg2rad(10.0); 0.0; 0.0];  # roll 10deg, target should be left
% camera_attitude = [deg2rad(-10.0); 0.0; 0.0];  # roll -10deg, target should be right

# rotate image frame to camera frame
R_if_cf = euler321(camera_mount(1), camera_mount(2), camera_mount(3));

# rotate camera frame to body planar frame
R_cf_bpf = euler321(camera_attitude(1), camera_attitude(2), camera_attitude(3));

# apply transforms - obtain target in body planar frame
target_bpf = R_cf_bpf * R_if_cf * target
