########################################################################################################################
#                                                                                                                      #
#                                          Test the rotational operator                                                #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 02.04.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
import pytest
from fdmoperators.pytorch_operators.rotational import scalar_rot as torch_rot
from fdmoperators.numpy_operators.rotational import scalar_rot as numpy_rot
from .misc import create_grid_pytorch, create_grid_numpy, compare_solutions


def test_scalar_rotational_pytorch():
    """ Test the PyTorch scalar rotational operator on an analytical case. """
    nchannels, nx, ny, dx, dy, X, Y = create_grid_pytorch()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = - Y ** 2
        field[channel, 1, :, :] = X ** 2
        analytical[channel, 0, :, :] = 2 * X + 2 * Y

    computed = torch_rot(field, dx, dy)

    compare_solutions(computed, analytical, atol=1e-14, rtol=1e-14)
    return X, Y, computed, analytical, field


def test_scalar_rotational_numpy():
    """ Test the NumPy scalar rotational operator on an analytical case. """
    nchannels, nx, ny, dx, dy, X, Y = create_grid_numpy()

    # Field and analytical solution initialisation
    field = np.zeros((2, ny, nx), dtype=np.float64)
    field[0, :, :] = - Y ** 2
    field[1, :, :] = X ** 2
    analytical = 2 * X + 2 * Y

    computed = numpy_rot(field, dx, dy)

    compare_solutions(computed, analytical, atol=1e-14, rtol=1e-14)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # torch_rot
    x, y, computed, analytical, field = test_scalar_rotational_pytorch()
    arrow_step = 5
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axarr.ravel()

    q = ax1.quiver(x[::arrow_step, ::arrow_step], y[::arrow_step, ::arrow_step],
                   field[0, 0, ::arrow_step, ::arrow_step], field[0, 1, ::arrow_step, ::arrow_step], pivot='mid')
    ax1.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')
    p1 = ax2.contourf(x, y, computed[0, 0], 100)
    fig.colorbar(p1, label='Computed scalar rotational', ax=ax2)
    p2 = ax3.contourf(x, y, analytical[0, 0], 100)
    fig.colorbar(p2, label='Analytical scalar rotational', ax=ax3)
    p3 = ax4.contourf(x, y, torch.abs(computed[0, 0] - analytical[0, 0]) / torch.abs(analytical[0, 0]), 100,
                      locator=ticker.LogLocator())
    fig.colorbar(p3, label='Relative difference', ax=ax4)
    plt.tight_layout()
    plt.savefig('test_scalar_rotational_pytorch.png')

    # numpy_rot
    x, y, computed, analytical, field = test_scalar_rotational_numpy()
    arrow_step = 5
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axarr.ravel()

    q = ax1.quiver(x[::arrow_step, ::arrow_step], y[::arrow_step, ::arrow_step],
                   field[0, ::arrow_step, ::arrow_step], field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax1.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')
    p2 = ax2.contourf(x, y, computed, 100)
    fig.colorbar(p2, label='Computed scalar rotational', ax=ax2)
    p3 = ax3.contourf(x, y, analytical, 100)
    fig.colorbar(p3, label='Analytical scalar rotational', ax=ax3)
    p4 = ax4.contourf(x, y, np.abs(computed - analytical) / np.abs(analytical), 100,
                      locator=ticker.LogLocator())
    fig.colorbar(p4, label='Relative difference', ax=ax4)
    plt.tight_layout()
    plt.savefig('test_scalar_rotational_numpy.png')
