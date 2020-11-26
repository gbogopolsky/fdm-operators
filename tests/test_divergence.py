########################################################################################################################
#                                                                                                                      #
#                                           Test the divergence operator                                               #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 28.02.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
import pytest
from fdmoperators.pytorch_operators.divergence import divergence as torch_div
from fdmoperators.numpy_operators.divergence import divergence as numpy_div
from .misc import create_grid_pytorch, create_grid_numpy, compare_solutions

# Tolerances for result comparisons
ATOL, RTOL = 1e-16, 1e-12


def test_divergence_pytorch():
    """ Test the PyTorch divergence operator on an analytical case. """
    nchannels, nx, ny, dx, dy, X, Y = create_grid_pytorch()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X ** 2
        field[channel, 1, :, :] = Y ** 2
        analytical[channel, 0, :, :] = 2 * X + 2 * Y

    computed = torch_div(field, dx, dy)

    compare_solutions(computed, analytical, atol=ATOL, rtol=RTOL)
    return X, Y, computed, analytical, field


@pytest.mark.parametrize("order", [2, 4])
def test_divergence_numpy(order):
    """ Test the NumPy operator on an analytical case. """
    nchannels, nx, ny, dx, dy, X, Y = create_grid_numpy()

    # Field and analytical solution initialisation
    field = np.zeros((2, ny, nx), dtype=np.float64)
    field[0, :, :] = X ** 2
    field[1, :, :] = Y ** 2
    analytical = 2 * X + 2 * Y

    computed = numpy_div(field, dx, dy, order=order)

    compare_solutions(computed, analytical, atol=ATOL, rtol=RTOL)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # torch_div
    x, y, computed, analytical, field = test_divergence_pytorch()
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()

    p1 = ax1.contourf(x, y, analytical[0, 0, :, :], 100)
    fig.colorbar(p1, label='Analytical divergence field', ax=ax1)
    p2 = ax2.contourf(x, y, computed[0, 0, :, :], 100)
    fig.colorbar(p2, label='Computed divergence field', ax=ax2)
    p3 = ax3.contourf(x, y, torch.abs(computed[0, 0, :, :] - analytical[0, 0, :, :]) / analytical[0, 0, :, :], 100,
                      locator=ticker.LogLocator())
    fig.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_divergence_pytorch.png')

    # numpy_div
    for order in [2, 4]:
        x, y, computed, analytical, field = test_divergence_numpy(order=order)
        fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
        ax1, ax2, ax3, _ = axarr.ravel()

        p1 = ax1.contourf(x, y, analytical, 100)
        fig.colorbar(p1, label='Analytical divergence field', ax=ax1)
        p2 = ax2.contourf(x, y, computed, 100)
        fig.colorbar(p2, label='Computed divergence field', ax=ax2)
        p3 = ax3.contourf(x, y, np.abs(computed - analytical) / np.abs(analytical), 100,
                          locator=ticker.LogLocator())
        fig.colorbar(p3, label='Relative difference', ax=ax3)
        plt.tight_layout()
        plt.savefig(f'test_divergence_numpy_order{order}.png')
