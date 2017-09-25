import numpy as np


class Transform:
    def __init__(self, **kwargs):
        self.frame_from = kwargs.get("from", None)
        self.frame_to = kwargs.get("to", None)

        self.R = np.zero((3, 3))
        self.t = np.zero((3, 1))

    def data(self):
        """ Obtain transform as a 4x4 homogeneous transform matrix """
        return np.array([[self.R[0][0], self.R[0][1], self.R[0][2], self.t[0]],
                         [self.R[1][0], self.R[1][1], self.R[1][2], self.t[1]],
                         [self.R[2][0], self.R[2][1], self.R[2][2], self.t[2]],
                         [0.0, 0.0, 0.0, 1.0]])

    def transform(self, x):
        """ Apply transform to x

        Args:

            x (np.array of size 3 or 4): vector to be transformed

        Returns:

            Transformed vector x

        """
        T = np.matrix(self.data)

        if len(x) != 4:
            x = np.array([x[0], x[1], x[2], 1.0])
            return (T * x)[0:3]
        else:
            return T * x
