import numpy as np


def rmse(predictions, ground_truth):
    """Root Mean Squared Error

    Parameters
    ----------
    predictions : np.array
        Predictions

    ground_truth : np.array
        Ground Truth

    Returns
    -------
    rmse : float
        Root Mean Squared Error

    """
    rmse = np.sqrt(((predictions - ground_truth)**2).mean())
    return rmse
