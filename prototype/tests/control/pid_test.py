#!/usr/bin/env python3
import unittest
from prototype.control.pid import PID


class PIDTest(unittest.TestCase):
    def test_init(self):
        pid = PID(1.0, 2.0, 3.0)

        self.assertEqual(pid.error_prev, 0.0)
        self.assertEqual(pid.error_sum, 0.0)

        self.assertEqual(pid.error_p, 0.0)
        self.assertEqual(pid.error_i, 0.0)
        self.assertEqual(pid.error_d, 0.0)

        self.assertEqual(pid.k_p, 1.0)
        self.assertEqual(pid.k_i, 2.0)
        self.assertEqual(pid.k_d, 3.0)

    def test_reset(self):
        pid = PID(1.0, 1.0, 1.0)

        pid.error_prev = 1.0
        pid.error_sum = 2.0

        pid.error_p = 1.0
        pid.error_i = 2.0
        pid.error_d = 3.0

        pid.reset()

        self.assertEqual(pid.error_prev, 0.0)
        self.assertEqual(pid.error_sum, 0.0)

        self.assertEqual(pid.error_p, 0.0)
        self.assertEqual(pid.error_i, 0.0)
        self.assertEqual(pid.error_d, 0.0)

    def test_update(self):
        pid = PID(1.0, 1.0, 1.0)

        setpoint = 0.0
        actual = 0.1
        dt = 0.1

        output = pid.update(setpoint, actual, dt)
        self.assertNotEqual(output, 0.0)
