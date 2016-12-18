#!/usr/bin/octave -qf
addpath("./lib/transform")
addpath("./lib/util")

robot_pos_ros = [1.0; 2.0; 3.0]
robot_att_euler = [deg2rad(10.0); deg2rad(20.0); deg2rad(30.0)];

% ROS (ENU) to Gazebo (NWU)
% ENU: x - right      y - forward     z - up
% NWU: x - forward    y - left        z - up
R_enu_nwu = [0.0, 1.0, 0.0;
             -1.0, 0.0, 0.0;
             0.0, 0.0, 1.0];

robot_quat = euler2quat(robot_att_euler(3),
                            robot_att_euler(2),
                            robot_att_euler(1),
                            "321");

[roll, pitch, yaw] = quat2euler(robot_quat(1),
                                robot_quat(2),
                                robot_quat(3),
                                robot_quat(4),
                                "321")

robot_rotmat = quat2rot(robot_quat(1),
                        robot_quat(2),
                        robot_quat(3),
                        robot_quat(4));

robot_pos_gaz = robot_rotmat * R_enu_nwu * robot_pos_ros


% Gazebo (NWU) to ROS (ENU)
R_nwu_enu = [0.0, -1.0, 0.0;
             1.0, 0.0, 0.0;
             0.0, 0.0, 1.0];

robot_pos_ros = R_nwu_enu * transpose(robot_rotmat) * robot_pos_gaz
