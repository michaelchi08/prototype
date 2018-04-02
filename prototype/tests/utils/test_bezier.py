import unittest

import numpy as np
import matplotlib.pylab as plt

from prototype.utils.bezier import bezier_quadratic
from prototype.utils.bezier import bezier_cubic
from prototype.utils.bezier import bezier_cubic_velocity
from prototype.utils.bezier import bezier_cubic_acceleration
from prototype.utils.bezier import bezier
from prototype.utils.bezier import decasteljau
from prototype.utils.bezier import bezier_derivative


class BezierTest(unittest.TestCase):
    def test_bezier_quadratic(self):
        # Setup anchor and control points
        P0 = np.array([0, 0, 0])
        C0 = np.array([0, 1, 0])
        P1 = np.array([1, 1, 0])

        # Setup book keeping
        t = 0.0
        dt = 0.01
        time = []
        track = []

        # Loop through Bezier curve
        while t < 1.0:
            s = bezier_quadratic(P0, C0, P1, t)
            time.append(t)
            track.append(s)
            t += dt

        # Convert list to np.array
        track = np.array(track)

        debug = True
        # debug = False
        if debug:
            # Plot Bezier curve
            plt.plot(P0[0], P0[1], color="red", marker="o",
                     label="Control Points")
            plt.plot(C0[0], C0[1], color="red", marker="o")
            plt.plot(P1[0], P1[1], color="red", marker="o")
            plt.plot(track[:, 0], track[:, 1], label="Bezier curve")
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            # -- Show plot
            plt.show()

    def test_bezier_cubic(self):
        # Setup anchor and control points
        P0 = np.array([0, 0, 0])
        C0 = np.array([0, 1, 0])
        C1 = np.array([1, 1, 3])
        P1 = np.array([1, 0, 3])

        # Setup book keeping
        t = 0.0
        dt = 0.01
        time = []
        track = []
        track_vel = []
        track_accel = []

        # Loop through Bezier curve
        while t < 1.0:
            s = bezier_cubic(P0, C0, C1, P1, t)
            v = bezier_cubic_velocity(P0, C0, C1, P1, t)
            a = bezier_cubic_acceleration(P0, C0, C1, P1, t)
            time.append(t)
            track.append(s)
            track_vel.append(v)
            track_accel.append(a)
            t += dt

        # Convert list to np.array
        track = np.array(track)
        track_vel = np.array(track_vel)
        track_accel = np.array(track_accel)

        est_p = [track[0, :]]
        est_v = track_vel[0, :] * dt
        # Simulate simple integration
        for k in range(1, len(time) - 1):
            est_v = est_v + track_vel[k] * dt
            est_p.append(est_v)
        est_p = np.array(est_p)

        debug = True
        # debug = False
        if debug:
            # Plot Bezier curve with its velocity and acceleration
            # -- Plot Bezier curve
            plt.subplot(311)
            plt.plot(P0[0], P0[1], color="red", marker="o",
                     label="Control Points")
            plt.plot(C0[0], C0[1], color="red", marker="o")
            plt.plot(C1[0], C1[1], color="red", marker="o")
            plt.plot(P1[0], P1[1], color="red", marker="o")
            plt.plot(track[:, 0], track[:, 1], label="Bezier curve")
            plt.plot(est_p[:, 0], est_p[:, 1], label="Estimate")
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.legend(loc=0)
            # -- Plot Bezier curve velocity
            plt.subplot(312)
            plt.plot(time, track_vel[:, 0], label="vx")
            plt.plot(time, track_vel[:, 1], label="vy")
            plt.plot(time, track_vel[:, 2], label="vz")
            plt.legend(loc=0)
            # -- Plot Bezier curve acceleration
            plt.subplot(313)
            plt.plot(time, track_accel[:, 0], label="x")
            plt.plot(time, track_accel[:, 1], label="y")
            plt.plot(time, track_accel[:, 2], label="z")
            plt.legend(loc=0)
            # -- Show plot
            plt.show()

    def test_bezier(self):
        # Setup anchor and control points
        P0 = np.array([0, 0, 0])
        C0 = np.array([0, 1, 0])
        C1 = np.array([1, 1, 3])
        P1 = np.array([1, 0, 3])

        # Setup book keeping
        t = 0.0
        dt = 0.01

        # Loop through Bezier curve
        while t < 1.0:
            s0 = bezier_cubic(P0, C0, C1, P1, t)
            s1 = bezier([P0, C0, C1, P1], t)
            t += dt
            self.assertTrue(np.allclose(s0, s1))

    def test_decasteljau(self):
        # Setup anchor and control points
        P0 = np.array([0, 0, 0])
        C0 = np.array([0, 1, 0])
        C1 = np.array([1, 1, 3])
        P1 = np.array([1, 0, 3])

        # Setup book keeping
        t = 0.0
        dt = 0.01

        # Loop through Bezier curve
        while t < 1.0:
            s0 = bezier_cubic(P0, C0, C1, P1, t)
            s1 = decasteljau([P0, C0, C1, P1], t)
            t += dt
            self.assertTrue(np.allclose(s0, s1))

    def test_bezier_derivative(self):
        # Setup anchor and control points
        P0 = np.array([0, 0, 0])
        C0 = np.array([1, 1, 1])
        C1 = np.array([2, 2, 2])
        P1 = np.array([3, 3, 3])

        # Check first bezier derivative
        expected = bezier_cubic_velocity(P0, C0, C1, P1, 0.5)
        observed = bezier_derivative([P0, C0, C1, P1], 0.5, 1)
        self.assertTrue(np.array_equal(expected, observed))

        # Check second bezier derivative
        expected = bezier_cubic_acceleration(P0, C0, C1, P1, 0.5)
        observed = bezier_derivative([P0, C0, C1, P1], 0.5, 2)
        self.assertTrue(np.array_equal(expected, observed))
