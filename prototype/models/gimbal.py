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
            np.array([0.045, 0.075, -0.085, 0.0, 0.0, 0.0])
        )

        # 6-dof transform from end-effector frame to dynamic camera
        self.tau_d = kwargs.get(
            "tau_d",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        # DH-params
        self.link = kwargs.get("link", np.array([0.0, 0.0, -0.02, 0.075]))

    def set_attitude(self, attitude):
        self.attitude = attitude
        self.link[0] = attitude[0]
        self.tau_d[3] = -attitude[1]

    def T_sb(self, tau_s):
        """ Form transform matrix from static camera to base_frame

        Parameters
        ----------
        tau_s : np.array
            Parameterization of the transform matrix where the first 3 elements
            in the vector is `t_G_sb`, the translation from static camera to
            base frame in global frame. Second 3 elements is `rpy_bs`, roll
            pitch yaw from static camera to base frame.

        Returns
        -------
        T_sb : np.array
            Transform matrix from static camera to base_frame

        """
        # Setup
        t_G_sb = tau_s[0:3]
        rpy_bs = tau_s[3:6]
        R_sb = euler2rot(rpy_bs, 321)

        # Create base frame
        T_sb = np.array([[R_sb[0, 0], R_sb[0, 1], R_sb[0, 2], t_G_sb[0]],
                         [R_sb[1, 0], R_sb[1, 1], R_sb[1, 2], t_G_sb[1]],
                         [R_sb[2, 0], R_sb[2, 1], R_sb[2, 2], t_G_sb[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        return T_sb

    def T_ed(self, tau_d):
        """ Form transform matrix from end-effector to dynamic camera

        Parameters
        ----------
        tau_s : np.array
            Parameterization of the transform matrix where the first 3 elements
            in the vector is `t_G_ed`, the translation from end effector to
            dynamic camera. Second 3 elements is `rpy_de`, roll pitch yaw from
            end effector to dynamic camera.

        Returns
        -------
        T_ed : np.array
            Transform matrix from end effector to dynamic camera

        """
        # Setup
        t_d_de = tau_d[0:3]
        R_ed = euler2rot(tau_d[3:6], 321)

        # Create transform
        T_ed = np.array([[R_ed[0, 0], R_ed[0, 1], R_ed[0, 2], t_d_de[0]],
                         [R_ed[1, 0], R_ed[1, 1], R_ed[1, 2], t_d_de[1]],
                         [R_ed[2, 0], R_ed[2, 1], R_ed[2, 2], t_d_de[2]],
                         [0.0, 0.0, 0.0, 1.0]])

        return T_ed

    def T_be(self, link):
        """ Form transform matrix from base_frame to end-effector

        Parameters
        ----------
        Lambda1 : float
            DH parameter (theta) for the first link
        w1 : np.array
            DH parameters for the first link (alpha, a, d)
        Lambda2 : float
            DH parameter (theta) for the second link
        w2 : np.array
            DH parameters for the second link (alpha, a, d)

        Returns
        -------
        T_be : np.array
            Transform matrix from base frame to end-effector

        """
        theta, alpha, a, d = self.link
        T_be = dh_transform(theta, alpha, a, d)
        return T_be

    def T_sd(self, tau_s, link, tau_d):
        # Transform from static camera to base frame
        T_sb = self.T_sb(tau_s)

        # Transform from base frame to end-effector
        T_be = self.T_be(link)

        # Transform from end-effector to dynamic camera
        T_ed = self.T_ed(tau_d)

        # Combine transforms
        T_se = dot(T_sb, T_be)  # Transform static camera to end effector
        T_sd = dot(T_se, T_ed)  # Transform static camera to dynamic camera

        return T_sd

    def calc_transforms(self):
        # Transform from static camera to base frame
        T_sb = self.T_sb(self.tau_s)

        # Transform from base frame to end-effector
        T_be = self.T_be(self.link)

        # Transform from end-effector to dynamic camera
        T_ed = self.T_ed(self.tau_d)

        # Create transforms
        T_se = dot(T_sb, T_be)  # Transform static camera to end effector
        T_sd = dot(T_se, T_ed)  # Transform static camera to dynamic camera

        # Create links
        links = [T_sb, T_se, T_sd]

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
