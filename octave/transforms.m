#!/usr/bin/octave -qf

addpath("./octave/lib/transform")
addpath("./octave/lib/util")

% q = euler2quat(0.1, 0.2, 0.3)
% [yaw, pitch, roll] = quat2euler(q)

target = [1.0; 0.0; 0.0]
camera_mount = [0.0; deg2rad(-90.0); 0.0]
R = euler321(camera_mount(1), camera_mount(2), camera_mount(3))
% R = euler123(camera_mount(1), camera_mount(2), camera_mount(3))

R * target
