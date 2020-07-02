########################################################################################################################
#                                                                                                                      #
#                                         Helper functions for the tests                                               #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np


# Grid parameters
xmin, xmax, ymin, ymax = 0, 1, 0, 1
nx, ny = 101, 101
nchannels = 10
dx, dy = (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1)


def create_grid_pytorch():
    """
    Initializes a square cartesian mesh for the PyTorch operators tests.

    Returns
    -------
    nchannels : int
        Number of channels

    nx, ny : int
        Number of elements

    dx, dy : float
        Step size

    X, Y : torch.Tensor
        Tensor containing the cartesian coordinates of size (ny, nx)
    """

    x, y = torch.linspace(xmin, xmax, nx, dtype=torch.float64), torch.linspace(ymin, ymax, ny, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x)  # Pay attention to the reversed order of the axes with torch.Tensor !

    return nchannels, nx, ny, dx, dy, X, Y


def create_grid_numpy():
    """
    Initializes a square cartesian mesh for the NumPy operators tests.

    Returns
    -------
    nchannels : int
        Number of channels

    nx, ny : int
        Number of elements

    dx, dy : float
        Step size

    X, Y : NumPy array
        Arrays containing the cartesian coordinates of size (nx, ny)
    """

    x, y = np.linspace(xmin, xmax, nx, dtype=np.float64), np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    return nchannels, nx, ny, dx, dy, X, Y
