########################################################################################################################
#                                                                                                                      #
#                                           Test the gradient operators                                                #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 02.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
import pytest
from pytorch_operators.gradient import gradient_scalar, gradient_diag, gradient_vector
from numpy_operators.gradient import gradient_scalar as gradient_scalar_numpy
from misc import create_grid_pytorch, create_grid_numpy


def test_gradient_scalar_pytorch():
    """ Test the PyTorch gradient_scalar operator on an analytical case. """
    nchannels, nx, ny, dx, dy, X, Y = create_grid_pytorch()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X ** 2 + Y ** 2
        analytical[channel, 0, :, :] = 2 * X
        analytical[channel, 1, :, :] = 2 * Y

    computed = gradient_scalar(field, dx, dy)

    assert torch.allclose(computed, analytical, atol=1e-13, rtol=1e-13)
    return X, Y, computed, analytical, field


def test_gradient_scalar_numpy():
    """ Test the NumPy gradient_scalar operator on an analytical case. """
    nchannels, nx, ny, dx, dy, X, Y = create_grid_numpy()

    # Field and analytical solution initialisation
    analytical = np.zeros((2, ny, nx), dtype=np.float64)
    field = X ** 2 + Y ** 2
    analytical[0, :, :] = 2 * X
    analytical[1, :, :] = 2 * Y

    computed = gradient_scalar_numpy(field, dx, dy)

    assert np.allclose(computed, analytical, atol=1e-13, rtol=1e-13)
    return X, Y, computed, analytical, field


def test_gradient_diag():
    """ Test the gradient_diag operator on an analytical case. The sum of the diagonal terms should be equal to the
        divergence of the input vector field. """
    # Create test grid
    nchannels, nx, ny, dx, dy, X, Y = create_grid_pytorch()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X ** 2
        field[channel, 1, :, :] = Y ** 2
        analytical[channel, 0, :, :] = 2 * X
        analytical[channel, 1, :, :] = 2 * Y

    computed = gradient_diag(field, dx, dy)

    assert torch.allclose(computed, analytical)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # gradient_scalar
    x, y, computed, analytical, field = test_gradient_scalar_pytorch()
    analytical_norm = torch.sqrt((analytical[0] ** 2).sum(0))
    computed_norm = torch.sqrt((computed[0] ** 2).sum(0))

    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()
    p1 = ax1.contourf(x, y, analytical_norm, 100)
    plt.colorbar(p1, label='Analytical gradient norm', ax=ax1)
    p2 = ax2.contourf(x, y, computed_norm, 100)
    plt.colorbar(p2, label='Computed gradient norm', ax=ax2)
    p3 = ax3.contourf(x, y, torch.abs(computed_norm - analytical_norm) / torch.abs(analytical_norm), 100,
                      locator=ticker.LogLocator())
    plt.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_gradient_scalar_pytorch.png')

    # gradient_scalar_numpy
    x, y, computed, analytical, field = test_gradient_scalar_numpy()
    analytical_norm = np.sqrt((analytical ** 2).sum(0))
    computed_norm = np.sqrt((computed ** 2).sum(0))

    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()
    p1 = ax1.contourf(x, y, analytical_norm, 100)
    plt.colorbar(p1, label='Analytical gradient norm', ax=ax1)
    p2 = ax2.contourf(x, y, computed_norm, 100)
    plt.colorbar(p2, label='Computed gradient norm', ax=ax2)
    p3 = ax3.contourf(x, y, np.abs(computed_norm - analytical_norm) / np.abs(analytical_norm), 100,
                      locator=ticker.LogLocator())
    plt.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_gradient_scalar_numpy.png')
