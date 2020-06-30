########################################################################################################################
#                                                                                                                      #
#                                                 Laplacian operator                                                   #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 30.06.2020                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np


def divergence(field, dx, dy, order=2):
    """
    Calculates the divergence of a vector field (second order accurate, decentered on the boundaries).

    Parameters
    ----------
    field : NumPy array
        Input 2D field of shape (2, H, W)

    dx, dy : float
        Spatial step for W and H directions

    order : integer
        2nd or 4th order inside the domain

    Returns
    -------
    NumPy array
        Output divergence with shape (H, W)
    """

    divergence = np.zeros_like(field)

    if order == 2:
        divergence[1:-1, 1:-1] = (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / (2 * dx) + \
                                 (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / (2 * dy)
    elif order == 4:
        divergence[1:-1, 1:-1] = (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / (2 * dx) + \
                                       (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / (2 * dy)
        divergence[2:-2, 2:-2] = (- field[0, 2:-2, 4:] + 8 * field[0, 2:-2, 3:-1]
                                  - 8 * field[0, 2:-2, 1:-3] + field[0, 2:-2, :-4]) / (12 * dx) + \
                                 (- field[1, 4:, 2:-2] + 8 * field[1, 3:-1, 2:-2]
                                  - 8 * field[1, 1:-3, 2:-2] + field[1, :-4, 2:-2]) / (12 * dy)

    # array sides except corners (respectively upper, lower, left and right sides)
    divergence[0, 1:-1] = (field[0, 0, 2:] - field[0, 0, :-2]) / (2 * dx) + \
                          (4 * field[1, 1, 1:-1] - 3 * field[1, 0, 1:-1] - field[1, 2, 1:-1]) / (2 * dy)
    divergence[-1, 1:-1] = (field[0, -1, 2:] - field[0, -1, :-2]) / (2 * dx) + \
                           (3 * field[1, -1, 1:-1] - 4 * field[1, -2, 1:-1] + field[1, -3, 1:-1]) / (2 * dy)
    divergence[1:-1, 0] = (4 * field[0, 1:-1, 1] - 3 * field[0, 1:-1, 0] - field[0, 1:-1, 2]) / \
                          (2 * dx) + (field[1, 2:, 0] - field[1, :-2, 0]) / (2 * dy)
    divergence[1:-1, -1] = (3 * field[0, 1:-1, -1] - 4 * field[0, 1:-1, -2] + field[0, 1:-1, -3]) / \
                           (2 * dx) + (field[1, 2:, -1] - field[1, :-2, -1]) / (2 * dy)

    # corners (respectively upper left, upper right, lower left and lower right)
    divergence[0, 0] = (4 * field[0, 0, 1] - 3 * field[0, 0, 0] - field[0, 0, 2]) / (2 * dx) + \
                       (4 * field[0, 1, 0] - 3 * field[0, 0, 0] - field[0, 2, 0]) / (2 * dy)
    divergence[-1, 0] = (4 * field[0, -1, 1] - 3 * field[0, -1, 0] - field[0, -1, 2]) / (2 * dx) + \
                        (3 * field[1, -1, 0] - 4 * field[1, -2, 0] + field[1, -3, 0]) / (2 * dy)
    divergence[0, -1] = (3 * field[0, 0, -1] - 4 * field[0, 0, -2] + field[0, 0, -3]) / (2 * dx) +\
                        (4 * field[1, 1, -1] - 3 * field[1, 0, -1] - field[1, 2, -1]) / (2 * dy)
    divergence[-1, -1] = (3 * field[0, -1, -1] - 4 * field[0, -1, -2] + field[0, -1, -3]) / (2 * dx) + \
                         (3 * field[1, -1, -1] - 4 * field[1, -2, -1] + field[1, -3, -1]) / (2 * dy)

    return divergence
