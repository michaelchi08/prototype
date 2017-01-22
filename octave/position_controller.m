#!/usr/bin/octave -qf
addpath('./lib/transform')
addpath('./lib/util')

setpoint = [1.0; 2.0; 3.0];
robot_pos = [0.0; 0.0; 0.0];

err = [setpoint(1) - robot_pos(1);
       setpoint(2) - robot_pos(2);
       setpoint(3) - robot_pos(3);];

err
R = euler123(0.0, 0.0, deg2rad(90.0));
err = R * err
