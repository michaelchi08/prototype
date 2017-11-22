import os
import shutil


def download(url, output_dir):
    """Donwload

    Parameters
    ----------
    url : str
        URL to download from
    output_dir : str
        Output directory

    Raises
    ------
    RuntimeError
        If cURL or Wget could not be found

    """
    # pre-check
    if os.path.isdir(output_dir) is False:
        raise RuntimeError("Invalid output dir [%s]" % output_dir)

    # build shell command string
    cmd = None
    if shutil.which("curl"):
        cmd = "cd {0} && curl -O {1}".format(output_dir, url)
    elif shutil.which("wget"):
        cmd = "cd {0} && wget {1}".format(output_dir, url)

    # run shell command
    if cmd:
        os.system(cmd)
    else:
        raise RuntimeError("Can't find cURL or Wget for downloading!")


def download_kitti_vo_dataset(output_dir):
    """Download KITTI VO dataset

    Parameters
    ----------
    output_dir : str
        Output directory

    """
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


def download_kitti_raw_dataset(output_dir):
    """Download KITTI RAW dataset

    Parameters
    ----------
    output_dir : str
        Output directory

    """
    pass
