import numpy as np


def mat2csv(output_file, data):
    """ Save matrix to file """
    np.savetxt(output_file, data, delimiter=",")
