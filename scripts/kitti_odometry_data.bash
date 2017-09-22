#!/bin/bash
set -e  # exit on first error

KITTI_URL="http://kitti.is.tue.mpg.de/kitti"
KITTI_FILES=( "data_odometry_calib.zip" \
              "data_odometry_gray.zip" \
              "data_odometry_poses.zip" )

mkdir -p /data
cd /data

for FILE in "${KITTI_FILES[@]}";
do
    echo "Downloading -> " $FILE;
    if [ ! -f $FILE ]; then
        wget $KITTI_URL/$FILE;
    fi

    unzip -o $FILE
done
