"""
BSD 3-Clause License

Copyright (c) 2017, JiaWang Bian
All rights reserved.

    Bian, JiaWang, et al. "Gms: Grid-based motion statistics for fast,
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


def draw_matches(src1, src2, kps1, kps2, matches):
    # Image width and height
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]

    # Setup output image
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    # Draw matches
    for i in range(len(matches)):
        left = kps1[matches[i].queryIdx].pt
        left = tuple(map(int, left))

        offset = np.array([src1.shape[1], 0])
        right = tuple(kps2[matches[i].trainIdx].pt + offset)
        right = tuple(map(int, right))

        cv2.line(output, left, right, (0, 255, 255))

    return output


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))


class GmsMatcher:
    def __init__(self, **kwargs):
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
        self.grid_size_right = Size(0, 0)
        self.grid_number_right = 0
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
        self.grid_size_left = Size(20, 20)
        self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.gms_matches = []

    def empty_matches(self):
        self.normalized_points1 = []
        self.normalized_points2 = []
        self.matches = []
        self.gms_matches = []

    # Normalize Key points to range (0-1)
    def normalize_keypoints(self, keypoints, size):
        npts = []

        for kp in keypoints:
            npts.append((kp.pt[0] / size.width,
                         kp.pt[1] / size.height))

        return npts

    # Normalize points to range (0-1)
    def normalize_points(self, points, size):
        npts = [(pt[0] / size.width, pt[1] / size.height) for pt in points]
        return npts

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches):
        v_matches = []

        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

        return v_matches

    def initialize_neighbours(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation):
        max_inlier = 0

        if not with_scale and not with_rotation:
            self.set_scale(0)
            max_inlier = self.run(1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type)
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
                num_inlier = self.run(rotation_type)
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
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_size_right.width = self.grid_size_left.width * self.scale_ratios[scale]
        self.grid_size_right.height = self.grid_size_left.height * self.scale_ratios[scale]
        self.grid_number_right = self.grid_size_right.width * self.grid_size_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_number_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_size_right)

    def run(self, rotation_type):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_number_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_number_left)]

            self.assign_match_pairs(GridType)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type):
        for i in range(self.matches_number):
            lp = self.normalized_points1[self.matches[i][0]]
            rp = self.normalized_points2[self.matches[i][1]]
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
        x = pt[0] * self.grid_size_left.width
        y = pt[1] * self.grid_size_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_size_left.width or y >= self.grid_size_left.height:
            return -1
        return x + y * self.grid_size_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_size_right.width))
        y = int(math.floor(pt[1] * self.grid_size_right.height))
        return x + y * self.grid_size_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_number_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_number_right)):
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

    def compute_matches(self, kps1, kps2, des1, des2, img1_shape, img2_shape=None):
        if self.gms_matches:
            self.empty_matches()

        # Image sizes
        img1_size = Size(img1_shape[1], img1_shape[0])
        img2_size = None
        if img2_shape is None:
            img2_size = img1_size
        else:
            img2_size = Size(img2_shape[1], img2_shape[0])

        # Match
        all_matches = self.matcher.match(des1, des2)
        self.normalized_points1 = self.normalize_keypoints(kps1, img1_size)
        self.normalized_points2 = self.normalize_keypoints(kps2, img2_size)
        self.matches_number = len(all_matches)
        self.matches = self.convert_matches(all_matches)
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)

        mask, num_inliers = self.get_inlier_mask(False, False)
        for i in range(len(mask)):
            if mask[i]:
                self.gms_matches.append(all_matches[i])

        return kps1, kps2, self.gms_matches


if __name__ == '__main__':
    img1 = cv2.imread("../data/nn_left.jpg")
    img2 = cv2.imread("../data/nn_right.jpg")

    img1 = imresize(img1, 480)
    img2 = imresize(img2, 480)

    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    gms = GmsMatcher(orb, matcher)

    matches = gms.compute_matches(img1, img2)
    gms.draw_matches(img1, img2)
