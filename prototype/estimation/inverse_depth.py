from math import cos
from math import sin
from math import atan2
from math import sqrt

import sympy
import numpy as np

from prototype.utils.quaternion.hamiltonian import quat2rot
from prototype.utils.quaternion.hamiltonian import quatmul
from prototype.utils.quaternion.hamiltonian import quatangle


def linearity_index_inverse_depth():
    """Linearity index of Inverse Depth Parameterization"""
    D, rho, rho_0, d_1, sigma_rho = sympy.symbols("D,rho,rho_0,d_1,sigma_rho")
    alpha = sympy.symbols("alpha")
    u = (rho * sympy.sin(alpha)) / (rho_0 * d_1 * (rho_0 - rho) + rho * sympy.cos(alpha))  # NOQA

    # first order derivative of u
    u_p = sympy.diff(u, rho)
    u_p = sympy.simplify(u_p)

    # second order derivative of u
    u_pp = sympy.diff(u_p, rho)
    u_pp = sympy.simplify(u_pp)

    # Linearity index
    L = (u_pp * 2 * sigma_rho) / (u_p)
    L = sympy.simplify(L)

    print()
    print("u: ", u)
    print("u': ", u_p)
    print("u'': ", u_pp)
    # print("L = ", L)
    print("L = ", L.subs(rho, 0))
    print()


def linearity_index_xyz_parameterization():
    """Linearity index of XYZ Parameterization"""
    d, d1, alpha, sigma_d = sympy.symbols("d,d1,alpha,sigma_d")
    alpha = sympy.symbols("alpha")
    u = (d * sympy.sin(alpha)) / (d1 + d * sympy.cos(alpha))

    # first order derivative of u
    u_p = sympy.diff(u, d)
    u_p = sympy.simplify(u_p)

    # second order derivative of u
    u_pp = sympy.diff(u_p, d)
    u_pp = sympy.simplify(u_pp)

    # Linearity index
    L = (u_pp * 2 * sigma_d) / (u_p)
    L = sympy.simplify(L)

    print()
    print("u: ", u)
    print("u': ", u_p)
    print("u'': ", u_pp)
    # print("L = ", L)
    print("L = ", L.subs(d, 0))
    print()


# def derive_J_jacobian():
#     r_W_C = sympy.symbols("r_W_C")
#
#     y =


def R(q):
    """Rotation matrix parameterized by a quaternion (w, x, y, z)

    Parameters
    ----------
    q : np.array
        Quaternion (w, x, y, z)

    Returns
    -------
    
        Rotation matrix (np.matrix)

    """
    return np.matrix(quat2rot(q))


def camera_motion_model(X, dt):
    """State update equation for the camera

    Parameters
    ----------
    X :
        camera state vector
    dt :
        Time difference (s)

    Returns
    -------
    
        Updated camera state vector

    """
    a_W = 0.0
    alpha_W = 0.0
    V_W_k = a_W * dt        # Linear velocity Gaussian process
    Omega_W = alpha_W * dt  # Angular velocity Guassian process

    # Expand state vector to its constituence
    r_W_C_k = X[0:3, 0].reshape(3, 1)  # Position
    q_WC_k = X[3:7, 0].reshape(4, 1)   # Orientation
    v_W_k = X[7:10, 0].reshape(3, 1)   # Linear velocity
    w_W_k = X[10:, 0].reshape(3, 1)    # Angular velocity

    # Update state
    r_W_C_kp1 = r_W_C_k + (v_W_k + V_W_k) * dt
    q_WC_kp1 = quatmul(q_WC_k, quatangle(w_W_k + Omega_W) * dt)
    v_W_kp1 = v_W_k + V_W_k
    w_W_kp1 = w_W_k + Omega_W

    return np.vstack([r_W_C_kp1, q_WC_kp1, v_W_kp1, w_W_kp1])


def h_C(y, r_W_C, q_WC, K, image_size):
    """Observation of a point from a camera location

    Parameters
    ----------
    y : np.array
        3D scene point vector
    r_W_C : np.array
        Camera position (x, y, z)
    q_WC : np.array
        Camera orientation quaternion (w, x, y, z)
    K : np.array
        Camera intrinsics matrix
    image_size : List of float
        Image size (image_width, image_height)

    Returns
    -------
    
        (u, v) pixel coordinates

    """
    # Expand 3D scene point vector y
    x, y, z, theta, psi, rho = y

    # Camera position from which the feature was first observed
    cam_0 = np.array([x, y, z]).reshape((3, 1))

    # Feature unit direction vector
    m = np.array([[cos(psi) * sin(theta)],
                  [-sin(psi)],
                  [cos(psi) * cos(theta)]])

    # Rotation matrix (world to camera) parameterized by quaternion
    R_CW = R(q_WC)

    # Feature observed from camera
    h = R_CW * (cam_0 + (1.0 / rho) * m)

    # Convert homogeneous coordinates to normalized coordinates
    u = h[0, 0] / h[2, 0]
    v = h[1, 0] / h[2, 0]

    # Apply camera calibration to convert normalized to pixel coordinates
    fx = K[0, 0]  # Focal length x-axis
    fy = K[1, 1]  # Focal length y-axis
    px = K[0, 2]  # Principle center x-axis
    py = K[1, 2]  # Principle center y-axis
    dx, dy = image_size  # Image size (image width, image height)

    u = px - fx / dx * u
    v = py - fy / dy * v

    return np.array([[u], [v]])


def feature_init(r_W_C, q_WC, pixel, d_min=1.0):
    """Feature initialization

    Parameters
    ----------
    r_W_C : np.array
        Camera position (x, y, z)
    q_WC : np.array
        Camera orientation quaternion (w, x, y, z)
    pixel : np.array
        Pixel coordinates
    d_min : float
        Predefined minimum depth (Default value = 1.0)

    Returns
    -------
    x, y, z
        Camera optical center where the 3D point was first observed
    theta
        Azimuth
    psi
        Elevation
    rho
        Inverse depth (1 / depth)

    """
    # initial min inverse depth (rho_0)
    rho_0 = 1.0 / d_min

    # Transform pixel observed from Camera frame to World frame
    point = np.array([[pixel[0]], [pixel[1]], [1.0]])
    h_W = R(q_WC) * point

    # Calculate the azimuth and elevation of the 3D scene point observed
    theta = atan2(-h_W[1], sqrt(h_W[0] ** 2 + h_W[2] ** 2))
    psi = atan2(h_W[0], h_W[2])

    return np.block([[r_W_C], [theta], [psi], [rho_0]])
