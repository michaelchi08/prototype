import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from prototype.utils.utils import rotz
from prototype.utils.utils import deg2rad
from prototype.utils.utils import euler2rot


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class QuadPlot:
    def __init__(self):
        self.arm_length = 0.4
        self.arm_config = "x"
        self.pose = [0.0, 0.0, 10.0,
                     deg2rad(10.0), deg2rad(20.0), deg2rad(30.0)]

    def plot(self, ax):
        # Quadrotor center
        center = np.array([0.0, 0.0, 0.0])

        # Quadrotor axis
        front = np.array([center[0] + (self.arm_length * 1.0),
                          center[1],
                          center[2]])
        left = np.array([center[0],
                         center[1] + (self.arm_length * 1.0),
                         center[2]])
        up = np.array([center[0],
                       center[1],
                       center[2] + (self.arm_length * 1.0)])

        # Quadrotor arms ("+" configuration)
        arm_l = np.array([center[0], center[1] + self.arm_length, center[2]])
        arm_r = np.array([center[0], center[1] - self.arm_length, center[2]])
        arm_f = np.array([center[0] + self.arm_length, center[1], center[2]])
        arm_b = np.array([center[0] - self.arm_length, center[1], center[2]])

        # Quadrotor arms ("x" configuration)
        if self.arm_config == "x":
            R_z = rotz(deg2rad(45.0))
            arm_l = np.dot(R_z, arm_l)
            arm_r = np.dot(R_z, arm_r)
            arm_f = np.dot(R_z, arm_f)
            arm_b = np.dot(R_z, arm_b)

        # Transform quadrotor
        R = euler2rot(self.pose[3:6], 123)
        t = np.array(self.pose[0:3])
        arm_l = np.dot(R, arm_l) + t
        arm_r = np.dot(R, arm_r) + t
        arm_f = np.dot(R, arm_f) + t
        arm_b = np.dot(R, arm_b) + t
        front = np.dot(R, front) + t
        left = np.dot(R, left) + t
        up = np.dot(R, up) + t
        center = np.dot(R, center) + t

        # Plot quadrotor
        ax.scatter(*center)

        ax.plot([center[0], front[0]],
                [center[1], front[1]],
                [center[2], front[2]], color="r")
        ax.plot([center[0], left[0]],
                [center[1], left[1]],
                [center[2], left[2]], color="g")
        ax.plot([center[0], up[0]],
                [center[1], up[1]],
                [center[2], up[2]], color="b")

        for arm in [arm_l, arm_r, arm_f, arm_b]:
            ax.plot([center[0], arm[0]],
                    [center[1], arm[1]],
                    [center[2], arm[2]],
                    color="black",
                    marker="o")
