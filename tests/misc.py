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
xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
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


def compare_solutions(computed, analytical, atol, rtol):
    """ Compare the two solutions with the given atol and rtol and raises an AssertionError with more info
    if a problem occurred. """
    if isinstance(computed, np.ndarray):
        lib = np
    elif isinstance(computed, torch.Tensor):
        lib = torch
    else:
        raise TypeError("Compared values are not of type 'torch.Tensor' or 'numpy.ndarray'.")

    try:
        assert lib.allclose(computed, analytical, atol=atol, rtol=rtol)
    except AssertionError as err:
        err_mask = lib.abs(computed - analytical) <= atol + rtol * lib.abs(analytical)
        # print("Masked arrays:")
        # print(computed[err_mask])
        # print(analytical[err_mask])
        # print("Differences in array:")
        # print(computed[err_mask] - analytical[err_mask])
        print("Max absolute error: {}".format((computed[err_mask] - analytical[err_mask]).max()))
        print("Max relative error: {}".format((computed[err_mask] - analytical[err_mask]).max()
                                              / analytical[err_mask].max()))
        raise err

