import unittest
import numpy as np
from prototype.control.carrot import CarrotController


class CarrotControllerTest(unittest.TestCase):
    def setUp(self):
        self.waypoints = np.array([[0.0, 0.0, 5.0],
                                   [5.0, 0.0, 5.0],
                                   [5.0, 5.0, 5.0],
                                   [0.0, 5.0, 5.0],
                                   [0.0, 0.0, 5.0]])
        self.look_ahead_dist = 0.1
        self.carrot_controller = CarrotController(self.waypoints,
                                                  self.look_ahead_dist)

    def test_init(self):
        self.assertTrue(np.array_equal(self.waypoints,
                                       self.carrot_controller.waypoints))
        self.assertTrue(np.array_equal(self.waypoints[0],
                                       self.carrot_controller.wp_start))
        self.assertTrue(np.array_equal(self.waypoints[1],
                                       self.carrot_controller.wp_end))
        self.assertEqual(self.look_ahead_dist,
                         self.carrot_controller.look_ahead_dist)

    def test_closest_point(self):
        wp_start = self.waypoints[0]
        wp_end = self.waypoints[1]
        point = np.array([0.0, 1.0, 5.0])

        result = self.carrot_controller.closest_point(wp_start,
                                                      wp_end,
                                                      point)
        closest_pt, retval = result
        closest_pt_expected = np.array([0.0, 0.0, 5.0])

        self.assertTrue(np.array_equal(closest_pt_expected, closest_pt))
        self.assertEqual(0, retval)

    def test_carrot_point(self):
        p_G = np.array([0.0, 0.0, 5.0])
        r = 0.1
        wp_start = self.waypoints[0]
        wp_end = self.waypoints[1]

        result = self.carrot_controller.carrot_point(p_G,
                                                     r,
                                                     wp_start,
                                                     wp_end)
        closest_pt, retval = result
        closest_pt_expected = np.array([0.1, 0.0, 5.0])

        self.assertTrue(np.array_equal(closest_pt_expected, closest_pt))
        self.assertEqual(0, retval)

    def test_update(self):
        p_G = np.array([0.0, 0.0, 5.0])
        closest_pt = self.carrot_controller.update(p_G)
        closest_pt_expected = np.array([0.1, 0.0, 5.0])
        self.assertEqual(1, self.carrot_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.0, 0.0, 5.0])
        closest_pt = self.carrot_controller.update(p_G)
        closest_pt_expected = np.array([5.1, 0.0, 5.0])
        self.assertEqual(1, self.carrot_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.1, 0.0, 5.0])
        closest_pt = self.carrot_controller.update(p_G)
        self.assertEqual(2, self.carrot_controller.wp_index)

        p_G = np.array([5.0, 4.0, 5.0])
        closest_pt = self.carrot_controller.update(p_G)
        closest_pt_expected = np.array([5.0, 4.1, 5.0])
        self.assertEqual(2, self.carrot_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.0, 5.0, 5.0])
        closest_pt = self.carrot_controller.update(p_G)
        closest_pt_expected = np.array([5.0, 5.1, 5.0])
        self.assertEqual(2, self.carrot_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        p_G = np.array([5.0, 5.1, 5.0])
        closest_pt = self.carrot_controller.update(p_G)
        closest_pt_expected = np.array([5.0, 5.2, 5.0])
        self.assertEqual(3, self.carrot_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))

        self.carrot_controller.wp_index = 4
        closest_pt = self.carrot_controller.update(p_G)
        closest_pt_expected = np.array([0.0, 0.0, 5.0])
        self.assertEqual(4, self.carrot_controller.wp_index)
        self.assertTrue(np.allclose(closest_pt_expected, closest_pt))
