import unittest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from prototype.viz.quadrotor import QuadPlot


def axis_equal_3dplot(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


class QuadPlotTest(unittest.TestCase):
    def test_plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        quad = QuadPlot()
        quad.plot(ax)

        axis_equal_3dplot(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
