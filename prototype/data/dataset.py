import os
import sys
import urllib.request


def download_reporthook(blocknum, blocksize, totalsize):
    sys.stdout.write(".")
    sys.stdout.flush()

def download(url, output_dir):
    if os.path.isdir(output_dir) is False:
        raise RuntimeError("{} invalid output dir!".format(output_dir))

    cmd = "cd {output_dir} && curl -O {url}".format(output_dir=output_dir,
                                                    url=url)
    os.system(cmd)


def download_kitti_vo_dataset(output_dir):
    base_url = "http://kitti.is.tue.mpg.de/kitti"
    files = ["data_odometry_calib.zip",
             "data_odometry_gray.zip",
             "data_odometry_poses.zip"]

    for f in files:
        url = os.path.join(base_url, f)
        print("Donwloading [{}]".format(f))
        print()
        download(url, output_dir)
        print()
