from math import sqrt
from math import pow

import unittest

from prototype.utils.gps import latlon_offset
from prototype.utils.gps import latlon_diff
from prototype.utils.gps import latlon_dist


class GPSTest(unittest.TestCase):
    def test_latlong_offset(self):
        # UWaterloo 110 yards Canadian Football field from one end to another
        lat = 43.474357
        lon = -80.550415
        offset_N = 44.1938
        offset_E = 90.2336

        # Calculate football field GPS coordinates
        lat_new, lon_new = latlon_offset(lat, lon, offset_N, offset_E)
        debug = False
        if debug:
            print("lat new: ", lat_new)
            print("lon new: ", lon_new)

        self.assertTrue(abs(43.474754 - lat_new) < 0.0015)
        self.assertTrue(abs(-80.549298 - lon_new) < 0.0015)

    def test_latlon_diff(self):
        # UWaterloo 110 yards Canadian Football field from one end to another
        lat_ref = 43.474357
        lon_ref = -80.550415
        lat = 43.474754
        lon = -80.549298

        dist_N = 0.0
        dist_E = 0.0

        # Calculate football field distance
        dist_N, dist_E = latlon_diff(lat_ref, lon_ref, lat, lon)
        dist = sqrt(pow(dist_N, 2) + pow(dist_E, 2))
        debug = False
        if debug:
            print("distance north: ", dist_N)
            print("distance east: ", dist_E)

        # 110 yards is approx 100 meters
        self.assertTrue(abs(100.0 - dist) < 1.0)

    def test_latlon_dist(self):
        # UWaterloo 110 yards Canadian Football field from one end to another
        lat_ref = 43.474357
        lon_ref = -80.550415
        lat = 43.474754
        lon = -80.549298

        # Calculate football field distance
        dist = latlon_dist(lat_ref, lon_ref, lat, lon)
        debug = False
        if debug:
            print("distance: ", dist)

        # 110 yards is approx 100 meters
        self.assertTrue(abs(100.0 - dist) < 1.0)
