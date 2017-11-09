import numpy as np
from numpy import eye
from numpy import dot
from numpy.linalg import inv


class KF(object):
    """Kalman Filter
    
    This class implements a generic Kalman Filter, the notation used in the code
    is baesd on:
    
        Thrun, S., Burgard, W., & Fox, D. (2006). Probabilistic robotics.
        Cambridge, Mass: The MIT Press.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, **kwargs):
        """ Constructor

        Args:

            mu (np.array): State vector
            S (np.array): Covariance matrix
            R (np.array): Motion noise matrix
            Q (np.array): Sensor noise matrix

        Note: the data input should be in the correct numpy array shape. For
        example, mu has to be of shape Nx1 where N is the number of states in
        the state vector.

            mu = numpy.array([0.0, 0.0, 0.0]).reshape(3, 1)

        """
        self.mu = kwargs["mu"]  # State vector
        self.S = kwargs["S"]    # Covariance matrix
        self.R = kwargs["R"]    # Motion noise matrix
        self.Q = kwargs["Q"]    # Sensor noise matrix

        self.S_p = None         # Predicted covariance matrix
        self.K = None           # Kalman gain

        # Convert S, R, Q to numpy matrix
        self.S = np.matrix(self.S)
        self.R = np.matrix(self.R)
        self.Q = np.matrix(self.Q)

    def prediction_update(self, A, B, u, dt):
        """Prediction update

        Parameters
        ----------
        A : np.array
            Transition matrix
        B : np.array
            Input matrix
        u : np.array
            Input
        dt : float
            Time difference

        Returns
        -------

        """
        A = np.matrix(A)
        B = np.matrix(B)
        self.mu_p = A * self.mu + B * u
        self.S_p = A * self.S * A.T + self.R

    def measurement_update(self, C, y):
        """Measurement update

        Parameters
        ----------
        C : np.array
            Measurement matrix
        y : np.array
            Measurement

        Returns
        -------

        """
        C = np.matrix(C)
        self.K = self.S_p * C.T * (C * self.S_p * C.T + self.Q).I
        self.mu = self.mu_p + self.K * (y - C * self.mu_p)
        self.S = (eye(len(self.mu)) - self.K * C) * self.S_p

    def estimate(self, u, y):
        """Estimate

        Parameters
        ----------
        u :
            
        y :
            

        Returns
        -------

        """
        self.prediction_update(u)
        self.measurement_update(y)
        return self.mu
