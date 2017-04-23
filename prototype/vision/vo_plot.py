#!/usr/bin/env python3
import csv

import numpy as np

import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.patches import PathPatch

DATASET_PATH = "/tmp/test"
FEATURE_DATA_FILE = "features.dat"


def load_features_data(dataset_path):
    csv_file = open(dataset_path + "/" + FEATURE_DATA_FILE, 'r')
    csv_reader = csv.reader(csv_file)

    data = []
    for line in csv_reader:
        f3d = np.array([float(line[0]), float(line[1]), float(line[2])])
        data.append(f3d)

    return np.vstack(data).T


def load_single_observation_data(fp):
    csv_file = open(fp, 'r')
    csv_reader = csv.reader(csv_file)

    data = {"time": None,
            "nb_observations": None,
            "state": None,
            "feature_2d": [],
            "feature_3d": []}

    data["time"] = float(next(csv_reader, None)[0])
    data["nb_observations"] = int(next(csv_reader, None)[0])
    state = next(csv_reader, None)
    data["state"] = np.array([float(state[0]),
                              float(state[1]),
                              float(state[2])])

    # parse observed features
    f_2d_line = True
    for line in csv_reader:
        if f_2d_line:
            f2d = np.array([float(line[0]), float(line[1])])
            data["feature_2d"].append(f2d)
            f_2d_line = False
        else:
            f3d = np.array([float(line[0]), float(line[1]), float(line[2])])
            data["feature_3d"].append(f3d)
            f_2d_line = True

    # convert list of vectors into 1 matrix, where each column is 1 observation
    data["feature_2d"] = np.vstack(data["feature_2d"]).T
    data["feature_3d"] = np.vstack(data["feature_3d"]).T

    return data


def load_all_observation_data(dataset_path):
    index_file = open(dataset_path + "/index.dat", 'r')
    observations = [line.strip() for line in index_file]

    data = {"time": [],
            "nb_observations": [],
            "state": [],
            "data": []}

    for f in observations:
        obs_data = load_single_observation_data(f)
        data["time"].append(obs_data["time"])
        data["nb_observations"].append(obs_data["nb_observations"])
        data["state"].append(obs_data["state"])
        data["data"].append(obs_data)

    return data


def plot_camera(ax):
    p = Rectangle((0, 0), 0.1, 0.1, ec="black", fill=False)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")


def plot_3d(fea_data, obs_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot 3d points
    ax.scatter(fea_data[0, :],
               fea_data[1, :],
               fea_data[2, :],
               c="r", s=5, depthshade=False)
    plt.show(block=False)

    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.0, 3.0])
    ax.set_zlim([-3.0, 3.0])

    lines = []
    for i in range(len(obs_data["data"])):
        obs = obs_data["data"][i]

        input("Time Step: " + str(i))
        print("State: " + str(obs["state"]))
        print()

        # ax.scatter(obs["feature_3d"][0, :],
        #            obs["feature_3d"][1, :],
        #            obs["feature_3d"][2, :],
        #            c="b", s=10, depthshade=False)

        for line in lines:
            try:
                line.remove()
            except ValueError:
                pass

        for j in range(obs["nb_observations"]):
            lines += ax.plot([obs["state"][0], obs["feature_3d"][0, j]],
                             [obs["state"][1], obs["feature_3d"][1, j]],
                             [0.0, obs["feature_3d"][2, j]], "b-")

        fig.canvas.draw()


if __name__ == "__main__":
    fea_data = load_features_data(DATASET_PATH)
    obs_data = load_all_observation_data(DATASET_PATH)
    plot_3d(fea_data, obs_data)
