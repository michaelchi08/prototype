import numpy as np


class IMUSim:
    """ """

    def __init__(self):
        # Previous velocity and angular velocity
        self.v_W_km1 = np.array([0.0, 0.0, 0.0])
        self.w_W_km1 = np.array([0.0, 0.0, 0.0])

        # Accelerometer and gyroscope bias
        self.b_a = 0.01
        self.b_g = 0.01

        # Guassian normal properties for accelerometer and gyroscope
        self.N_a_sigma = 0.1
        self.N_g_sigma = 0.1

        # IMU output
        self.a_W = np.array([0.0, 0.0, 0.0])
        self.w_W = np.array([0.0, 0.0, 0.0])

    def update(self, v_W, w_W, dt):
        """Update IMU

        Parameters
        ----------
        v_W : np.array of size 3
            Velocity in world frame
        w_W : np.array of size 3
            Angular velocity in world frame
        dt : float
            Time difference

        Returns
        -------
        a_W
            IMU acceleration and gyroscope

        """
        # Generate new gaussian normal noise
        N_a = np.random.normal(0.0, self.N_a_sigma ** 2, 3)
        N_g = np.random.normal(0.0, self.N_g_sigma ** 2, 3)

        # Update IMU biases
        self.b_a += np.random.normal(0.0, self.N_a_sigma ** 2, 3)
        self.b_g += np.random.normal(0.0, self.N_g_sigma ** 2, 3)

        # Update IMU output
        self.a_W = self.a_W + (v_W - self.v_W_km1) * dt + self.b_a * dt + N_a
        self.w_W = self.w_W + (w_W - self.w_W_km1) * dt + self.b_g * dt + N_g

        # Update
        self.v_W_km1 = v_W
        self.w_W_km1 = w_W

        return (self.a_W, self.w_W)


def generate_signal(n, dt, q_white, q_walk, q_ramp, random_state=0):
    """

    Parameters
    ----------
    n :

    dt :

    q_white :

    q_walk :

    q_ramp :

    random_state :
         (Default value = 0)

    Returns
    -------

    """
    rng = np.random.RandomState(random_state)
    white = q_white ** 0.5 * rng.randn(n) * dt
    walk = q_walk ** 0.5 * dt * np.cumsum(rng.randn(n))
    ramp = q_ramp * dt * np.arange(n)
    return white + walk * dt + ramp * dt
