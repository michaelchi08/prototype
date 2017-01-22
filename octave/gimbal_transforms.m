#!/usr/bin/octave -qf
addpath('./lib/transform')
addpath('./lib/util')

target_bpf = [0.0; 0.0; -10.0]
frame_if = [0.0, deg2rad(10.0), 0.0];
R = euler123(frame_if(1), frame_if(2), frame_if(3))

target = R * target_bpf

dist = norm(target);
roll_setpoint = asin(target(2) / dist)
pitch_setpoint = -asin(target(1) / dist)
