#!/bin/bash
DOWNLOAD_PATH=/data/kitti/raw
KITTI_RAW_URL="http://kitti.is.tue.mpg.de/kitti/raw_data"

# Setup download path
mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH

# Download
if [ ! -d 2011_09_26 ]; then
    if [ ! -f 2011_09_26_calib.zip ]; then
        curl -O $KITTI_RAW_URL/2011_09_26_calib.zip
    fi
    unzip 2011_09_26_calib.zip

    if [ ! -f 2011_09_26_drive_0005_sync ]; then
        curl -O $KITTI_RAW_URL/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip
    fi
    unzip 2011_09_26_drive_0005_sync.zip
fi
