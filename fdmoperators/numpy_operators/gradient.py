########################################################################################################################
#                                                                                                                      #
#                                                 Gradient operator                                                    #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 30.06.2020                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np


def gradient_scalar(field, dx, dy):
    """
    Calculates the gradient of a scalar field (second order accurate, degraded to first order on the boundaries).

    Parameters
    ----------
    field : NumPy array
        Input 1D field of shape (H, W)

    dx, dy : float
        Spatial step for W and H directions

    Returns
    -------
    NumPy array
        Output gradient with shape (2, H, W)
    """
    gradient = np.zeros((2, *field.shape), dtype=field.dtype)

    gradient[0, :, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    gradient[0, :, 0] = (4 * field[:, 1] - 3 * field[:, 0] - field[:, 2]) / (2 * dx)
    gradient[0, :, -1] = - (4 * field[:, -2] - 3 * field[:, -1] - field[:, -3]) / (2 * dx)

    gradient[1, 1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy)
    gradient[1, 0, :] = (4 * field[1, :] - 3 * field[0, :] - field[2, :]) / (2 * dy)
    gradient[1, -1, :] = - (4 * field[-2, :] - 3 * field[-1, :] - field[-3, :]) / (2 * dy)

    return gradient
