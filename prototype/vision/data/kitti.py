import os

import cv2
import numpy as np

from prototype.utils.filesystem import walkdir
from prototype.vision.vo.vo import BasicVO


def get_scale(ground_truth, frame_id):
    """

    Parameters
    ----------
    ground_truth :

    frame_id :


    Returns
    -------

    """
    # Obtain prev pose
    pose_prev = ground_truth[frame_id - 1]
    x_prev, y_prev, z_prev = pose_prev

    # Obtain pose
    pose = ground_truth[frame_id]
    x, y, z = pose

    # Calculate scale
    dx = (x - x_prev)
    dy = (y - y_prev)
    dz = (z - z_prev)
    scale = np.sqrt(dx * dx + dy * dy + dz * dz)

    return scale


def benchmark_mono_vo(data_path, sequence, vo, **kwargs):
    """Benchmark Monocular VO

    Parameters
    ----------
    data_path : str
        Path to data
    sequence : str
        Data sequence
    vo :

    """
    map_size = kwargs.get("map_size", (600, 600))
    visualize = kwargs.get("visualize", False)

    # Setup
    img_dir = os.path.join(data_path, sequence, "image_0")
    img_files, nb_imgs = parse_data_dir(img_dir)
    ground_truth = parse_ground_truth(data_path, sequence)
    vo = BasicVO(718.8560, 607.1928, 185.2157)

    # Create trajectory map
    traj_map = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)

    # Iterate throught different images
    for img_id in range(nb_imgs):
        # Load image
        img_fname = str(img_id).zfill(6) + '.png'
        img_path = os.path.join(data_path, sequence, 'image_0', img_fname)
        img = cv2.imread(img_path, 0)

        # Perform visual odometry
        scale = get_scale(ground_truth, img_id)
        est_R, est_t = vo.update(img_id, img, scale)
        if img_id > 2:
            est_x = est_t[0]
            est_z = est_t[2]
        else:
            est_x = 0.0
            est_z = 0.0

        draw_x = int(est_x) + 290
        draw_y = int(est_z) + 90
        x = int(ground_truth[img_id][0]) + 290
        y = int(ground_truth[img_id][2]) + 90

        # Visualize
        if visualize:
            cv2.rectangle(traj_map, (10, 20), (600, 60), (0, 0, 0), -1)
            cv2.circle(traj_map, (x, y), 1, (0, 0, 255), 1)
            cv2.circle(traj_map, (draw_x, draw_y), 1, (0, 255, 0), 1)

            cv2.imshow('Road facing camera', img)
            cv2.imshow('Trajectory', traj_map)
            cv2.waitKey(1)
