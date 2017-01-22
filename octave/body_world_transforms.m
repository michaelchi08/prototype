#!/usr/bin/octave -qf
addpath('./lib/transform')
addpath('./lib/util')


% target in inertial frame
target_pos = [1.0; 2.0; 0.0]

% quad in inertial frame
quad_pos = [0.0; 0.0; 3.0]
quad_euler = [0.0; 0.0; deg2rad(90.0)];

% inertial to body frame
quat = euler2quat(quad_euler(1), quad_euler(2), quad_euler(3), "123");
R = quat2rot(quat(1), quat(2), quat(3), quat(4));
target_pos_bf = R * (target_pos - quad_pos)

% body to inertial frame
target_pos = (inverse(R) * target_pos_bf) + quad_pos
