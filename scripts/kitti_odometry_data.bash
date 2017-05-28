#!/bin/bash
set -e  # exit on first error

mkdir /data
cd /data
unzip -o data_odometry_calib.zip
unzip -o data_odometry_gray.zip
unzip -o data_odometry_poses.zip
