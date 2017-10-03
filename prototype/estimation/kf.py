import numpy as np
from numpy import eye
from numpy import dot
from numpy.linalg import inv


class KF(object):
    """ Kalman Filter

    This class implements a generic Kalman Filter, the notation used in the code
    is baesd on:

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

        self.A = None
        self.B = None
        self.C = None

        self.R = None
        self.Q = None

        self.S_p = None
        self.K = None

        # Convert S, R, Q to numpy matrix
        self.S = np.matrix(self.S)
        self.R = np.matrix(self.R)
        self.Q = np.matrix(self.Q)

    def prediction_update(self, u, dt):
        """ Prediction update

        Args:

            g (np.array): Process model
            G (np.array): Derivative of process model
            dt (float): Time difference

        """
        self.mu_p = dot(self.A, self.mu) + dot(self.B, u)
        self.S_p = dot(self.A, dot(self.S, self.A.T)) + self.R
        self.K = self.S_p * self.C * (self.C * self.S_p * self.C.T).I + self.Q

    def measurement_update(self, y):
        """ Measurement update

        Args:

            y (np.array): Measurement
            h (np.array): Measurement model
            H (np.array): Derivative of measurement model
            dt (float): Time difference

        """
        self.mu = self.mu_p + self.K * (y - self.C * self.mu_p)
        self.S = (eye(len(self.A)) - self.K * self.C) * self.S_p

    # def estimate(self, u, y):
    #     self.prediction_update(u)
    #     self.measurement_update(y)
