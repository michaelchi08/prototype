import numpy as np
from numpy import dot

from prototype.utils.utils import deg2rad
from prototype.utils.euler import euler2rot

# from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def axis_equal_3dplot(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                        for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_3d_cube(ax, width, origin, orientation):
    """ Plot 3D cube

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Plot axes

    width : float
        Cube width

    origin : np.array
        Origin

    orientation : np.array
        Orientation

    """
    # Cube points
    points = np.array([[-1, -1, -1],
                       [1, -1, -1],
                       [1, 1, -1],
                       [-1, 1, -1],
                       [-1, -1, 1],
                       [1, -1, 1],
                       [1, 1, 1],
                       [-1, 1, 1]])
    points = points * (width / 2.0)

    # Rotate Cube
    R = euler2rot([deg2rad(i) for i in orientation], 123)
    points = np.array([dot(R, pt) for pt in points])

    # Translate Cube
    points += origin

    # Plot mesh grid
    r = [-1, 1]
    X, Y = np.meshgrid(r, r)

    # List of sides' polygons of figure
    verts = [[points[0], points[1], points[2], points[3]],
             [points[4], points[5], points[6], points[7]],
             [points[0], points[1], points[5], points[4]],
             [points[2], points[3], points[7], points[6]],
             [points[1], points[2], points[6], points[5]],
             [points[4], points[7], points[3], points[0]],
             [points[2], points[3], points[7], points[6]]]

    # Plot sides
    ax.add_collection3d(Poly3DCollection(verts,
                                         facecolors="black",
                                         linewidths=1,
                                         edgecolors="red",
                                         alpha=0.25))
