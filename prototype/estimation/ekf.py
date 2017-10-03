import numpy as np
from numpy import eye


class EKF(object):
    """ Extended Kalman Filter

    This class implements a generic EKF filter, the notation used in the code is
    baesd on:

        Thrun, S., Burgard, W., & Fox, D. (2006). Probabilistic robotics.
        Cambridge, Mass: The MIT Press.

    """
    def __init__(self, **kwargs):
        """ Constructor

        Args:

            mu (np.array - size Nx1): State vector
            S (np.array - size NxN): Covariance matrix
            R (np.array - size NxN): Motion noise matrix
            Q (np.array - size NxN): Sensor noise matrix

        Note: the data input should be in the correct numpy array shape. For
        example, mu has to be of shape Nx1 where N is the number of states in
        the state vector.

            mu = numpy.array([0.0, 0.0, 0.0]).reshape(3, 1)

        """
        self.mu = kwargs["mu"]  # State vector
        self.S = kwargs["S"]    # Covariance matrix
        self.R = kwargs["R"]    # Motion noise matrix
        self.Q = kwargs["Q"]    # Sensor noise matrix

        self.mu_p = None
        self.S_p = None
        self.K = None

        # Convert S, R, Q to numpy matrix
        self.S = np.matrix(self.S)
        self.R = np.matrix(self.R)
        self.Q = np.matrix(self.Q)

    def prediction_update(self, u, g, G, dt):
        """ Prediction update

        Args:

            u (np.array): Input
            g (np.array): Process model
            G (np.array): Derivative of process model
            dt (float): Time difference

        """
        self.mu_p = g
        self.S_p = G * self.S * G.T + self.R

    def measurement_update(self, y, h, H):
        """ Measurement update

        Args:

            y (np.array): Measurement
            h (np.array): Measurement model
            H (np.array): Derivative of measurement model
            dt (float): Time difference

        """
        K = self.S_p * H.T * (H * self.S_p * H.T).I + self.Q
        self.mu = self.mu_p + K * (y - h)
        self.S = (eye(len(self.mu)) - K * H) * self.S_p
