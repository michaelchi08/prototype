import numpy as np


def mat2csv(output_file, data):
    """Save matrix to file

    Parameters
    ----------
    output_file : str
        Output file path
    data : np.array
        Target data to save

    Returns
    -------

    """
    np.savetxt(output_file, data, delimiter=",")
