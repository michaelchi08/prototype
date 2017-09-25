from prototype.estimation.ekf import EKF
from prototype.utils.utils import deg2rad
from prototype.models.omni_wheel import OmniWheel


class EKFTest(unittest.TestCase):
    # Setup
    dt = 0.1
    mu = [0.0, 0.0, 0.0]
    R = np.matrix([[0.05**2, 0.0, 0.0],
                   [0.0, 0.05**2, 0.0],
                   [0.0, 0.0, deg2rad(0.5)**2]])
    Q = np.matrix([[0.5^2, 0.0, 0.0],
                   [0.0, 0.5^2, 0.0],
                   [0.0, 0.0, deg2rad(10)^2]])
    S = eye(3)
    ekf = ekf_setup(dt, mu, R, Q, S)

    # Simulation parameters
    t_end = 10
    T = 0:dt:t_end
    x_0 = [0.0, 0.0, 0.0]
    x = zeros(len(x_0), len(T))
    x(:, 1) = x_0
    y = zeros(len(ekf.Q), len(T))
    u = [0.0 -10.0 10.0]

    # Storge
    mu = zeros(len(x_0))
    noise = zeros(len(ekf.R), len(x_0))

    # Simulation
    for t in T:
        # Update state
        e = gaussian_noise(ekf.R)
        x(:, t) =  g(u, x(:, t - 1), dt) + e
        noise(:, t) = e

        # Take measurement
        d = gaussian_noise(ekf.Q)
        y(:, t) = x(:, t) + d

        # EKF
        [ekf] = ekf.prediction_update(ekf, @g, @G, u, 0)
        [ekf] = ekf.measurement_update(ekf, @h, @H, y(:, t), 0)

        # store ekf results
        mu(:, t) = ekf.mu

    # plot_trajectory_2d(T, x, y, mu)
