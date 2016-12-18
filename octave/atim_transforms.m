#!/usr/bin/octave -qf
addpath('./lib/transform')
addpath('./lib/util')

# target position in camera frame (NED)
tgt = [1.0; 0.0; 0.0]

# camera facing downwards (i.e. pitch -90 deg in NED)
cam_mnt = [0.0; deg2rad(-90.0); 0.0];

# camera attitude in inertial frame
cam_att = [0.0; deg2rad(10.0); 0.0];  # pitch 10, target should be ahead
% cam_att = [0.0; deg2rad(-10.0); 0.0];  # pitch -10, target should be behind
% cam_att = [deg2rad(10.0); 0.0; 0.0];  # roll 10, target should be left
% cam_att = [deg2rad(-10.0); 0.0; 0.0];  # roll -10, target should be right

# rotate image frame (if) to camera frame (cf)
# !!IMPORTANT NOTE!!: This is dangerous by setting pitch to -90 we are in
# [GIMBAL LOCK]! In our case it is ok because we only want to point camera
# downwards, but if roll and pitch mount angles are introduced this rotation
# matrix will not work!
R_if_cf = euler321(cam_mnt(1), cam_mnt(2), cam_mnt(3));

# rotate camera frame (cf) to body planar frame (bpf)
R_cf_bpf = euler321(cam_att(1), cam_att(2), cam_att(3));

# apply transforms - obtain target in body planar frame (bpf)
tgt_bpf = R_cf_bpf * R_if_cf * tgt
