import csv

import numpy as np

import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
# from matplotlib.patches import PathPatch

DATASET_PATH = "/tmp/test"
LANDMARK_DATA_FILE = "features.dat"


def load_feature_data(dataset_path):
    # setup
    csv_file = open(dataset_path + "/" + LANDMARK_DATA_FILE, 'r')
    csv_reader = csv.reader(csv_file)

    # parse features file
    data = []
    for line in csv_reader:
        f3d = np.array([float(line[0]), float(line[1]), float(line[2])])
        data.append(f3d)

    # clean up
    csv_file.close()

    return np.vstack(data).T


def load_single_observed_data(fp):
    # setup
    csv_file = open(fp, 'r')
    csv_reader = csv.reader(csv_file)

    data = {
        "time": None,
        "nb_observations": None,
        "state": None,
        "keypoints": [],
        "feature_ids": []
    }

    # parse time, nb_observations and robot state
    data["time"] = float(next(csv_reader, None)[0])
    x = next(csv_reader, None)
    data["state"] = np.array([float(x[0]), float(x[1]), float(x[2])])
    data["nb_observations"] = int(next(csv_reader, None)[0])

    # parse observed features
    keypoint_line = True
    for line in csv_reader:
        if keypoint_line:
            keypoint = np.array([float(line[0]), float(line[1])])
            data["keypoints"].append(keypoint)
            keypoint_line = False
        else:
            feature_id = int(line[0])
            data["feature_ids"].append(feature_id)
            keypoint_line = True

    # convert list of vectors into 1 matrix, where each column is 1 observation
    data["keypoints"] = np.vstack(data["keypoints"]).T

    # clean up
    csv_file.close()

    return data


def load_observed_data(dataset_path):
    # setup
    index_file = open(dataset_path + "/index.dat", 'r')
    observations = [line.strip() for line in index_file]

    # load
    data = {"time": [],
            "nb_observations": [],
            "state": [],
            "data": []}

    for f in observations:
        obs_data = load_single_observed_data(f)
        data["time"].append(obs_data["time"])
        data["nb_observations"].append(obs_data["nb_observations"])
        data["state"].append(obs_data["state"])
        data["data"].append(obs_data)

    # clean up
    index_file.close()

    return data


def plot_camera(ax):
    p = Rectangle((0, 0), 0.1, 0.1, ec="black", fill=False)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")


def plot_3d(feature_data, observed_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot 3d points
    ax.scatter(
        feature_data[0, :],
        feature_data[1, :],
        feature_data[2, :],
        c="r",
        s=5,
        depthshade=False)
    plt.show(block=False)

    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.0, 3.0])
    ax.set_zlim([-3.0, 3.0])

    lines = []
    for i in range(len(observed_data["data"])):
        obs = observed_data["data"][i]

        input("Time Step: " + str(i))
        print("State: " + str(obs["state"]))
        print("Number of features: " + str(obs["nb_observations"]))
        print("Landmark IDs: " + str(obs["feature_ids"]))
        print()

        for line in lines:
            try:
                line.remove()
            except ValueError:
                pass

        for j in range(obs["nb_observations"]):
            feature_id = obs["feature_ids"][j]
            feature = feature_data[:, feature_id]
            lines += ax.plot([obs["state"][0], feature[0]],
                             [obs["state"][1], feature[1]],
                             [0.0, feature[2]], "b-")

        fig.canvas.draw()
