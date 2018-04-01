import unittest
import numpy as np
from prototype.control.quadrotor.waypoint import WaypointController

from prototype.utils.utils import rad2deg
from prototype.utils.utils import deg2rad
from prototype.utils.euler import euler2rot


class WaypointControllerTest(unittest.TestCase):
    def setUp(self):
        self.waypoints = np.array([[0.0, 0.0, 5.0],
                                   [5.0, 0.0, 5.0],
                                   [5.0, 5.0, 5.0],
                                   [0.0, 5.0, 5.0],
                                   [0.0, 0.0, 5.0]])
        self.look_ahead_dist = 0.01
        self.wp_controller = WaypointController(self.waypoints,
                                                self.look_ahead_dist)

    def test_init(self):
        self.assertTrue(np.array_equal(self.waypoints,
                                       self.wp_controller.waypoints))
        self.assertTrue(np.array_equal(self.waypoints[0],
                                       self.wp_controller.wp_start))
        self.assertTrue(np.array_equal(self.waypoints[1],
                                       self.wp_controller.wp_end))
        self.assertEqual(self.look_ahead_dist,
                         self.wp_controller.look_ahead_dist)

    def test_closest_point(self):
        wp_start = self.waypoints[0]
        wp_end = self.waypoints[1]
        point = np.array([0.0, 1.0, 5.0])

        result = self.wp_controller.closest_point(wp_start, wp_end, point)
        closest_pt, retval = result
        closest_pt_expected = np.array([0.0, 0.0, 5.0])

        self.assertTrue(np.array_equal(closest_pt_expected, closest_pt))
        self.assertEqual(0, retval)

    def test_carrot_point(self):
        p_G = np.array([0.0, 0.0, 5.0])
        r = 0.1
        wp_start = self.waypoints[0]
        wp_end = self.waypoints[1]

        result = self.wp_controller.carrot_point(p_G,
                                                 r,
                                                 wp_start,
                                                 wp_end)
        closest_pt, retval = result
        closest_pt_expected = np.array([0.1, 0.0, 5.0])

        self.assertTrue(np.array_equal(closest_pt_expected, closest_pt))
        self.assertEqual(0, retval)

    def test_calc_yaw_to_waypoint(self):
        wp = np.array([1.0, 1.0, 3.0])
        pos = np.array([0.0, 0.0, 3.0])

        heading = self.wp_controller.calc_yaw_to_waypoint(wp, pos)
        self.assertEqual(heading, deg2rad(45.0))
        # print("wp", wp.tolist())
        # print("pos", pos.tolist())
        # print("heading", rad2deg(heading))

    def test_world_to_body(self):
        wp = np.array([1.0, 1.0, 3.0])
        pos = np.array([0.0, 0.0, 4.0])
        yaw = deg2rad(45.0)
        R = euler2rot(np.array([0.0, 0.0, yaw]), 321)

        # print("")
        # print("wp: ", wp)
        # print("pos: ", pos)
        # print(np.dot(R.T, (wp - pos)))

    def test_update_waypoint(self):
        p_G = np.array([0.0, 0.0, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([0.1, 0.0, 5.0])
        self.assertEqual(1, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.0, 0.0, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([5.1, 0.0, 5.0])
        self.assertEqual(1, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.1, 0.0, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        self.assertEqual(2, self.wp_controller.wp_index)

        p_G = np.array([5.0, 4.0, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([5.0, 4.1, 5.0])
        self.assertEqual(2, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.0, 5.0, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([5.0, 5.1, 5.0])
        self.assertEqual(2, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.0, 5.1, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([5.0, 5.2, 5.0])
        self.assertEqual(3, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        self.wp_controller.wp_index = 4
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([0.0, 0.0, 5.0])
        self.assertEqual(4, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

    def test_update(self):
        p_G = np.array([0.0, 0.0, 5.0])
        closest_pt = self.wp_controller.update_waypoint(p_G)
        closest_pt_expected = np.array([0.1, 0.0, 5.0])
        self.assertEqual(1, self.wp_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))
