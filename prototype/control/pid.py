#!/usr/bin/env python3


class PID(object):
    """ PID controller """

    def __init__(self, k_p, k_i, k_d):
        self.error_prev = 0.0
        self.error_sum = 0.0

        self.error_p = 0.0
        self.error_i = 0.0
        self.error_d = 0.0

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def reset(self):
        """ Reset PID controller """
        self.error_prev = 0.0
        self.error_sum = 0.0

        self.error_p = 0.0
        self.error_i = 0.0
        self.error_d = 0.0

    def update(self, setpoint, actual, dt):
        """ Update PID controller """
        # update errors
        error = setpoint - actual
        self.error_sum += error * dt

        # update output
        self.error_p = self.k_p * error
        self.error_i = self.k_i * self.error_sum
        self.error_d = self.k_d * (error - self.error_prev) / dt
        output = self.error_p + self.error_i + self.error_d

        # update error
        self.error_prev = error

        return output
