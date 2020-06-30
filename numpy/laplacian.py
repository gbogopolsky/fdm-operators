########################################################################################################################
#                                                                                                                      #
#                                                 Laplacian operator                                                   #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 30.06.2020                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np


def laplacian(field, dx, dy, order=2, b=0):
    """
    Calculates the laplacian of a scalar field (second order accurate, decentered on the boundaries).
    The output shape is the same as the input shape.

    Parameters
    ----------
    field : NumPy array
        Input 1D field of shape (H, W)

    dx, dy : float
        Spatial step for W and H directions

    order : integer
        2nd or 4th order inside the domain

    b : float
        Parameter for the discretisation shape (see Hirsch p.164)

    Returns
    -------
    Numpy array
        Output laplacian with shape (H, W)
    """

    laplacian = np.zeros_like(field)

    laplacian[1:-1, 1:-1] = (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2 + \
                            (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2
    if order == 2:
        laplacian[1:-1, 1:-1] = (1 - b) * ((field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2 +
                                (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2) + \
                            b * (field[2:, 2:] + field[2:, :-2] + field[:-2, :-2] + field[:-2, 2:] - 4 * field[1:-1, 1:-1]) \
                            / (2 * dx**2)
    elif order == 4:
        laplacian[2:-2, 2:-2] = (- field[4:, 2:-2] + 16 * field[3:-1, 2:-2] - 30 * field[2:-2, 2:-2] + 16 * field[1:-3, 2:-2]
                                 - field[:-4, 2:-2]) / (12 * dy**2) + \
                                (- field[2:-2, 4:] + 16 * field[2:-2, 3:-1] - 30 * field[2:-2, 2:-2] + 16 * field[2:-2, 1:-3]
                                 - field[2:-2, :-4]) / (12 * dx**2)
        laplacian[1, 1:-1] = (field[2, 1:-1] + field[0, 1:-1] - 2 * field[1, 1:-1]) / dy**2 + \
                             (field[1, 2:] + field[1, :-2] - 2 * field[1, 1:-1]) / dx**2
        laplacian[-2, 1:-1] = (field[-1, 1:-1] + field[-3, 1:-1] - 2 * field[-2, 1:-1]) / dy**2 + \
                              (field[-2, 2:] + field[-2, :-2] - 2 * field[-2, 1:-1]) / dx**2
        laplacian[1:-1, 1] = (field[2:, 1] + field[:-2, 1] - 2 * field[1:-1, 1]) / dy**2 + \
                             (field[1:-1, 2] + field[1:-1, 0] - 2 * field[1:-1, 1]) / dx**2
        laplacian[1:-1, -2] = (field[2:, -2] + field[:-2, -2] - 2 * field[1:-1, -2]) / dy**2 + \
                              (field[1:-1, -1] + field[1:-1, -3] - 2 * field[1:-1, -2]) / dx**2

    laplacian[0, 1:-1] = \
        (2 * field[0, 1:-1] - 5 * field[1, 1:-1] + 4 * field[2, 1:-1] - field[3, 1:-1]) / dy**2 + \
        (field[0, 2:] + field[0, :-2] - 2 * field[0, 1:-1]) / dx**2
    laplacian[-1, 1:-1] = \
        (2 * field[-1, 1:-1] - 5 * field[-2, 1:-1] + 4 * field[-3, 1:-1] - field[-4, 1:-1]) / dy**2 + \
        (field[-1, 2:] + field[-1, :-2] - 2 * field[-1, 1:-1]) / dx**2
    laplacian[1:-1, 0] = \
        (field[2:, 0] + field[:-2, 0] - 2 * field[1:-1, 0]) / dy**2 + \
        (2 * field[1:-1, 0] - 5 * field[1:-1, 1] + 4 * field[1:-1, 2] - field[1:-1, 3]) / dx**2
    laplacian[1:-1, -1] = \
        (field[2:, -1] + field[:-2, -1] - 2 * field[1:-1, -1]) / dy**2 + \
        (2 * field[1:-1, -1] - 5 * field[1:-1, -2] + 4 * field[1:-1, -3] - field[1:-1, -4]) / dx**2

    # corners (respectively upper left, upper right, lower left and lower right)
    laplacian[0, 0] = \
        (2 * field[0, 0] - 5 * field[1, 0] + 4 * field[2, 0] - field[3, 0]) / dy**2 + \
        (2 * field[0, 0] - 5 * field[0, 1] + 4 * field[0, 2] - field[0, 3]) / dx**2
    laplacian[0, -1] = \
        (2 * field[0, -1] - 5 * field[1, -1] + 4 * field[2, -1] - field[3, -1]) / dy**2 + \
        (2 * field[0, -1] - 5 * field[0, -2] + 4 * field[0, -3] - field[0, -4]) / dx**2
    laplacian[-1, 0] = \
        (2 * field[-1, 0] - 5 * field[-2, 0] + 4 * field[-3, 0] - field[-4, 0]) / dy**2 + \
        (2 * field[-1, 0] - 5 * field[-1, 1] + 4 * field[-1, 2] - field[-1, 3]) / dx**2
    laplacian[-1, -1] = \
        (2 * field[-1, -1] - 5 * field[-2, -1] + 4 * field[-3, -1] - field[-4, -1]) / dy**2 + \
        (2 * field[0, -1] - 5 * field[0, -2] + 4 * field[0, -3] - field[0, -4]) / dx**2

    return laplacian
