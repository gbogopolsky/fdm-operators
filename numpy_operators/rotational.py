########################################################################################################################
#                                                                                                                      #
#                                                 Rotational operator                                                  #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 30.06.2020                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np


def scalar_rot(field, dx, dy):
    """
    Calculates the scalar rotational of a 2D vector field (second order accurate).

    Parameters
    ----------
    field : NumPy array
        Input 2D field of shape (2, H, W)

    dx, dy : float
        Spatial step for W and H directions

    Returns
    -------
    NumPy array
        Output rotational with shape (H, W)
    """
    rotational = np.zeros(field.shape[1:], dtype=field.dtype)

    # first compute dfield_y / dx
    rotational[:, 1:-1] = (field[1, :, 2:] - field[1, :, :-2]) / (2 * dx)
    rotational[:, 0] = (4 * field[1, :, 1] - 3 * field[1, :, 0] - field[1, :, 2]) / (2 * dx)
    rotational[:, -1] = (3 * field[1, :, -1] - 4 * field[1, :, -2] + field[1, :, -3]) / (2 * dx)

    # second compute dfield_x / dy
    rotational[1:-1, :] -= (field[0, 2:, :] - field[0, :-2, :]) / (2 * dy)
    rotational[0, :] -= (4 * field[0, 1, :] - 3 * field[0, 0, :] - field[0, 2, :]) / (2 * dy)
    rotational[-1, :] -= (3 * field[0, -1, :] - 4 * field[0, -2, :] + field[0, -3, :]) / (2 * dy)

    return rotational
