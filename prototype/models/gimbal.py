from math import cos
from math import sin
from math import pi

import numpy as np
from numpy import dot

from prototype.utils.euler import euler2rot


def dh_transform(theta, alpha, a, d):
    """ Denavit–Hartenberg transform matrix

    Parameters
    ----------
    theta : float
        Angle (radians)
    alpha : float
        Angle (radians)
    a : float
        Offset (m)
    d : float
        Offset (m)

    Returns
    -------
    DH Transform matrix

    """
    c = cos
    s = sin

    return np.array([
        [c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
        [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
        [0.0, s(alpha), c(alpha), d],
        [0.0, 0.0, 0.0, 1.0],
    ])


class GimbalModel:
    def __init__(self, **kwargs):
        self.attitude = np.array([0.0, 0.0])

        # 6-dof transform from static camera to base mechanism frame
        self.tau_s = kwargs.get(
            "tau_s",
            np.array([-0.045, -0.085, 0.08, 0.0, 0.0, 0.0])
        )

        # 6-dof transform from end-effector frame to dynamic camera
        self.tau_d = kwargs.get(
            "tau_d",
            np.array([0.0, 0.0, 0.0, pi / 2.0, 0.0, -pi / 2.0])
        )

        # DH-params (theta, alpha, a, d)
        self.link1 = kwargs.get("link1", np.array([0.0, pi / 2.0, 0.1, 0.0]))
        self.link2 = kwargs.get("link2", np.array([0.0, 0.0, 0.0, 0.0]))

    def set_attitude(self, attitude):
        self.attitude = attitude
        # self.link1[0] = attitude[0]
        # self.tau_d[3] = -attitude[1]

    def T_bs(self, tau_s):
        """ Form transform matrix from static camera to base mechanism

        Parameters
        ----------
        tau_s : np.array
            Parameterization of the transform matrix where the first 3 elements
            in the vector is the translation from static camera to base frame.
            Second 3 elements is rpy, which is the roll pitch yaw from static
            camera to base frame.

        Returns
        -------
        T_bs : np.array
            Transform matrix from static camera to base_frame

        """
        # Setup
        t = tau_s[0:3]
        rpy = tau_s[3:6]
        R = euler2rot(rpy, 321)

        # Create base frame
        T_bs = np.array([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                         [R[1, 0], R[1, 1], R[1, 2], t[1]],
                         [R[2, 0], R[2, 1], R[2, 2], t[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        return T_bs

    def T_de(self, tau_d):
        """ Form transform matrix from end-effector to dynamic camera

        Parameters
        ----------
        tau_s : np.array
            Parameterization of the transform matrix where the first 3 elements
            in the vector is the translation from end effector to dynamic
            camera frame. Second 3 elements is rpy, which is the roll pitch yaw
            from from end effector to dynamic camera frame.

        Returns
        -------
        T_de : np.array
            Transform matrix from end effector to dynamic camera

        """
        # Setup
        t = tau_d[0:3]
        R = euler2rot(tau_d[3:6], 321)

        # Create transform
        T_de = np.array([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                         [R[1, 0], R[1, 1], R[1, 2], t[1]],
                         [R[2, 0], R[2, 1], R[2, 2], t[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        return T_de

    def T_eb(self, link1, link2):
        """ Form transform matrix from base_frame to end-effector

        Parameters
        ----------
        link1 : np.array
            DH parameters (theta, alpha, a, d) for the first link
        link2 : np.array
            DH parameters (theta, alpha, a, d) for the second link

        Returns
        -------
        T_eb : np.array
            Transform matrix from base frame to end-effector

        """
        theta1, alpha1, a1, d1 = link1
        theta2, alpha2, a2, d2 = link2
        T_1b = np.linalg.inv(dh_transform(theta1, alpha1, a1, d1))
        T_e1 = np.linalg.inv(dh_transform(theta2, alpha2, a2, d2))
        return np.dot(T_e1, T_1b)

    def T_ds(self):
        # Transform from static camera to base frame
        T_bs = self.T_bs(self.tau_s)

        # Transform from base frame to end-effector
        T_eb = self.T_eb(self.link1, self.link2)

        # Transform from end-effector to dynamic camera
        T_de = self.T_de(self.tau_d)

        # Create transforms
        T_es = dot(T_eb, T_bs)  # Transform static camera to end effector
        T_ds = dot(T_de, T_es)  # Transform static camera to dynamic camera

        return T_ds

    def calc_transforms(self):
        # Transform from static camera to base frame
        T_bs = self.T_bs(self.tau_s)

        # Transform from base frame to end-effector
        T_eb = self.T_eb(self.link1, self.link2)

        # Transform from end-effector to dynamic camera
        T_de = self.T_de(self.tau_d)

        # Create links
        links = [T_bs, T_eb, T_de]

        return links

    def save(self, output_path):
        """ Save gimbal configuration as a yaml file

        Parameters
        ----------
        output_path : str
            Output path for gimbal configuration

        """
        config = """
tau_s : [{tau_s_tx}, {tau_s_ty}, {tau_s_tz}, {tau_s_roll}, {tau_s_pitch}, {tau_s_yaw}]
tau_d : [{tau_d_tx}, {tau_d_ty}, {tau_d_tz}, {tau_d_roll}, {tau_d_pitch}, {tau_d_yaw}]
link : [{link_theta}, {link_alpha}, {link_a}, {link_d}]
        """.format(
            # tau_s
            tau_s_tx=self.tau_s[0],
            tau_s_ty=self.tau_s[1],
            tau_s_tz=self.tau_s[2],
            tau_s_roll=self.tau_s[3],
            tau_s_pitch=self.tau_s[4],
            tau_s_yaw=self.tau_s[5],
            # tau_d
            tau_d_tx=self.tau_d[0],
            tau_d_ty=self.tau_d[1],
            tau_d_tz=self.tau_d[2],
            tau_d_roll=self.tau_d[3],
            tau_d_pitch=self.tau_d[4],
            tau_d_yaw=self.tau_d[5],
            # link
            link_theta=self.link[0],
            link_alpha=self.link[1],
            link_a=self.link[2],
            link_d=self.link[3]
        )

        config_file = open(output_path, "w")
        config_file.write(config.strip())
        config_file.close()
