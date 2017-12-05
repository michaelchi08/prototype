"""
BSD 3-Clause License

Copyright (c) 2017, JiaWang Bian
All rights reserved.

    Bian, JiaWang, et al. "GMS: Grid-based motion statistics for fast,
    ultra-robust feature correspondence." 2017 IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR). IEEE, 2017.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import math
from enum import Enum

import cv2
import numpy as np

THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [[1, 2, 3,
                      4, 5, 6,
                      7, 8, 9],
                     [4, 1, 2,
                      7, 5, 3,
                      8, 9, 6],
                     [7, 4, 1,
                      8, 5, 2,
                      9, 6, 3],
                     [8, 7, 4,
                      9, 5, 1,
                      6, 3, 2],
                     [9, 8, 7,
                      6, 5, 4,
                      3, 2, 1],
                     [6, 9, 8,
                      3, 5, 7,
                      2, 1, 4],
                     [3, 6, 9,
                      2, 5, 8,
                      1, 4, 7],
                     [2, 3, 6,
                      1, 5, 9,
                      4, 7, 8]]


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))


def draw_matches(img0, img1, kp0, kp1, matches, display_type=0):
    height = max(img0.shape[0], img1.shape[0])
    width = img0.shape[1] + img1.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:img0.shape[0], 0:img0.shape[1]] = img0
    output[0:img1.shape[0], img0.shape[1]:] = img1[:]

    if display_type == 0:
        for i in range(len(matches)):
            left = kp0[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp1[matches[i].trainIdx].pt, (img0.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif display_type == 1:
        for i in range(len(matches)):
            left = kp0[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp1[matches[i].trainIdx].pt, (img0.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp0[matches[i].queryIdx].pt
            for x in zip(kp1[matches[i].trainIdx].pt, (img0.shape[1], 0)):
                right = tuple(sum(x))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    cv2.imshow('show', output)
    cv2.waitKey()


class GmsMatcher:
    def __init__(self):
        self.scale_ratios = [1.0, 1.0 / 2,
                             1.0 / math.sqrt(2.0),
                             math.sqrt(2.0), 2.0]

        # Normalized vectors of 2D points
        self.normalized_points1 = []
        self.normalized_points2 = []

        # Matches - list of pairs representing numbers
        self.matches = []
        self.matches_number = 0

        # Grid Size
        self.grid_sz_right = Size(0, 0)
        self.grid_nb_right = 0
        # x      : left grid idx
        # y      :  right grid idx
        # value  : how many matches from idx_left to idx_right
        self.motion_statistics = []

        self.number_of_points_per_cell_left = []
        # Inldex  : grid_idx_left
        # Value   : grid_idx_right
        self.cell_pairs = []

        # Every Matches has a cell-pair
        # first  : grid_idx_left
        # second : grid_idx_right
        self.match_pairs = []

        # Inlier Mask for output
        self.inlier_mask = []
        self.grid_neighbor_right = []

        # Grid initialize
        self.grid_sz_left = Size(20, 20)
        self.grid_nb_left = self.grid_sz_left.width * self.grid_sz_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_nb_left, 9))

    # Normalize key points to range (0-1)
    def normalize_points(self, kp, size):
        npts = []
        for keypoint in kp:
            npts.append((keypoint.pt[0] / size.width,
                         keypoint.pt[1] / size.height))

        return npts

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches, v_matches):
        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

    def initialize_neighbours(self, neighbor, grid_sz):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_sz)

    def get_nb9(self, idx, grid_sz):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_sz.width
        idx_y = idx // grid_sz.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_sz.width or idx_yy < 0 or idx_yy >= grid_sz.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_sz.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation, npts0, npts1):
        max_inlier = 0

        if not with_scale and not with_rotation:
            self.set_scale(0)
            max_inlier = self.run(1, npts0, npts1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type, npts0, npts1)
                    if num_inlier > max_inlier:
                        vb_inliers = self.inlier_mask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        elif with_scale and not with_rotation:
            vb_inliers = []
            for rotation_type in range(1, 9):
                num_inlier = self.run(rotation_type, npts0, npts1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                num_inlier = self.run(1, npts0, npts1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_sz_right.width = self.grid_sz_left.width * self.scale_ratios[scale]
        self.grid_sz_right.height = self.grid_sz_left.height * self.scale_ratios[scale]
        self.grid_nb_right = self.grid_sz_right.width * self.grid_sz_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_nb_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_sz_right)

    def run(self, rotation_type, npts0, npts1):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        # self.motion_statistics = np.zeros((int(self.grid_nb_left), int(self.grid_nb_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_nb_left), int(self.grid_nb_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_nb_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_nb_left)]

            self.assign_match_pairs(GridType, npts0, npts1)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type, npts0, npts1):
        for i in range(self.matches_number):
            lp = npts0[self.matches[i][0]]
            rp = npts1[self.matches[i][1]]
            lgidx = self.match_pairs[i][0] = self.get_grid_index_left(lp, grid_type)

            if grid_type == 1:
                rgidx = self.match_pairs[i][1] = self.get_grid_index_right(rp)
            else:
                rgidx = self.match_pairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.motion_statistics[int(lgidx)][int(rgidx)] += 1
            self.number_of_points_per_cell_left[int(lgidx)] += 1

    def get_grid_index_left(self, pt, type_of_grid):
        x = pt[0] * self.grid_sz_left.width
        y = pt[1] * self.grid_sz_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_sz_left.width or y >= self.grid_sz_left.height:
            return -1
        return x + y * self.grid_sz_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_sz_right.width))
        y = int(math.floor(pt[1] * self.grid_sz_right.height))
        return x + y * self.grid_sz_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_nb_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_nb_right)):
                value = self.motion_statistics[i]
                if value[j] > max_number:
                    self.cell_pairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.cell_pairs[i]
            nb9_lt = self.grid_neighbor_left[i]
            nb9_rt = self.grid_neighbor_right[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[current_rotation_pattern[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.motion_statistics[int(ll), int(rr)]
                thresh += self.number_of_points_per_cell_left[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            if score < thresh:
                self.cell_pairs[i] = -2

    def compute_matches(self, kp0, kp1, des0, des1, matches, img0, img1=None):
        # Obtain image sizes
        size0 = Size(img0.shape[1], img0.shape[0])
        size1 = size0
        if img1 is not None:
            size1 = Size(img1.shape[1], img1.shape[0])

        npts0 = self.normalize_points(kp0, size0)
        npts1 = self.normalize_points(kp1, size1)
        self.matches_number = len(matches)
        self.convert_matches(matches, self.matches)
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_sz_left)

        mask, num_inliers = self.get_inlier_mask(False, False, npts0, npts1)

        final_matches = []
        for i in range(len(mask)):
            if mask[i]:
                final_matches.append(matches[i])

        return final_matches
