#!/usr/bin/octave -qf
addpath('./lib/transform')
addpath('./lib/util')

pkg load symbolic

% % GIMBAL SETPOINTS
% target_bpf = [0.0; 0.0; -10.0]
% frame_if = [0.0, deg2rad(10.0), 0.0];
% R = euler123(frame_if(1), frame_if(2), frame_if(3))
% target = R * target_bpf
%
% dist = norm(target);
% roll_setpoint = asin(target(2) / dist)
% pitch_setpoint = -asin(target(1) / dist)

% % AprilTag measurements in C to P
% target_C = [0; 0; 10]
% gimbal_I = [0; deg2rad(-10.0); 0.0]
%
% C_R_N = [
%     0, 0, 1;
%     -1, 0, 0;
%     0, -1, 0;
% ];
% N_R_G = euler321(gimbal_I(1), gimbal_I(2), gimbal_I(3));
% G_R_P = [
%     0, 0, 1;
%     0, 1, 0;
%     -1, 0, 0;
% ]
%
% target_N = C_R_N * target_C
% target_G = N_R_G * C_R_N * target_C
% target_bpf = G_R_P * N_R_G * C_R_N * target_C


% SENSITIVITY ANALYSIS
syms x y z roll pitch yaw

target_C = [x; y; z];
C_R_N = [
    0, 0, 1;
    -1, 0, 0;
    0, -1, 0;
];
N_R_G = euler321(roll, pitch, yaw);
G_R_P = [
    0, 0, 1;
    0, 1, 0;
    -1, 0, 0;
];

% target_N = C_R_N * target_C
% target_G = N_R_G * C_R_N * target_C
target_bpf = G_R_P * N_R_G * C_R_N * target_C


jacobian(target_bpf)
