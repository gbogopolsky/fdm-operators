########################################################################################################################
#                                                                                                                      #
#                                           Test the laplacian operator                                                #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 28.02.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
import pytest
from fdmoperators.pytorch_operators.laplacian import laplacian as torch_lapl
from fdmoperators.numpy_operators.laplacian import laplacian as numpy_lapl
from .misc import create_grid_pytorch, create_grid_numpy, compare_solutions


@pytest.mark.parametrize("b", [0, 1/3, 1])
def test_laplacian_pytorch(b):
    """ Test the PyTorch laplacian operator on a polynomial profile."""
    nchannels, nx, ny, dx, dy, X, Y = create_grid_pytorch()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X ** 3 + Y ** 3
        analytical[channel, 0, :, :] = 6 * X + 6 * Y

    computed = torch_lapl(field, dx, dy, b=b)
    print(f'b = {b} : ', torch.sum(torch.abs(computed[0, 0, :, :] - analytical[0, 0, :, :])))

    compare_solutions(computed, analytical, atol=1e-10, rtol=1e-11)
    return X, Y, computed, analytical, field


@pytest.mark.parametrize("b", [0, 1/3, 1])
def test_laplacian_pytorch_exp(b):
    """ Test the PyTorch laplacian operator on an exponential profile."""
    nchannels, nx, ny, dx, dy, X, Y = create_grid_pytorch()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = torch.exp(X) + torch.exp(2 * Y)
        analytical[channel, 0, :, :] = torch.exp(X) + 4 * torch.exp(2 * Y)

    computed = torch_lapl(field, dx, dy, b=b)
    print(f'b = {b} : ', torch.sum(torch.abs(computed[0, 0, :, :] - analytical[0, 0, :, :])))

    compare_solutions(computed, analytical, atol=1e-10, rtol=1e-3)
    return X, Y, computed, analytical, field


@pytest.mark.parametrize("b", [0, 1/3, 1])
@pytest.mark.parametrize("order", [2, 4])
def test_laplacian_numpy(b, order):
    """ Test the NumPy laplacian operator on a polynomial profile."""
    nchannels, nx, ny, dx, dy, X, Y = create_grid_numpy()

    # Field and analytical solution initialisation
    field = X ** 3 + Y ** 3
    analytical = 6 * X + 6 * Y

    computed = numpy_lapl(field, dx, dy, order=order, b=b)
    print(f'b = {b} : ', np.sum(np.abs(computed - analytical)))

    compare_solutions(computed, analytical, atol=1e-10, rtol=1e-11)
    return X, Y, computed, analytical, field


@pytest.mark.parametrize("b", [0, 1/3, 1])
@pytest.mark.parametrize("order", [2, 4])
def test_laplacian_numpy_exp(b, order):
    """ Test the NumPy laplacian operator on an exponential profile."""
    nchannels, nx, ny, dx, dy, X, Y = create_grid_numpy()

    # Field and analytical solution initialisation
    field = np.exp(X) + np.exp(2 * Y)
    analytical = np.exp(X) + 4 * np.exp(2 * Y)

    computed = numpy_lapl(field, dx, dy, order=order, b=b)
    print(f'b = {b} : ', np.sum(np.abs(computed - analytical)))

    compare_solutions(computed, analytical, atol=1e-12, rtol=1e-3)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # torch_lapl
    def pytorch_plot(test_function, case):
        x, y, computed, analytical, _ = test_function(b=0)
        fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
        ax1, ax2, ax3, _ = axarr.ravel()

        p1 = ax1.contourf(x, y, analytical[0, 0, :, :], 100)
        fig.colorbar(p1, label='Analytical laplacian field', ax=ax1)
        p2 = ax2.contourf(x, y, computed[0, 0, :, :], 100)
        fig.colorbar(p2, label='Computed laplacian field', ax=ax2)
        p3 = ax3.contourf(x, y, torch.abs(computed[0, 0, :, :] - analytical[0, 0, :, :]) / analytical[0, 0, :, :], 100,
                          locator=ticker.LogLocator())
        fig.colorbar(p3, label='Relative difference', ax=ax3)
        plt.tight_layout()
        plt.savefig(f'test_laplacian_pytorch_{case}.png')
    pytorch_plot(test_laplacian_pytorch, "poly")
    pytorch_plot(test_laplacian_pytorch_exp, "exp")

    # numpy_lapl
    def numpy_plot(test_function, case):
        for order in [2, 4]:
            x, y, computed, analytical, _ = test_function(order=order, b=0)
            fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
            ax1, ax2, ax3, _ = axarr.ravel()

            p1 = ax1.contourf(x, y, analytical, 100)
            fig.colorbar(p1, label='Analytical laplacian field', ax=ax1)
            p2 = ax2.contourf(x, y, computed, 100)
            fig.colorbar(p2, label='Computed laplacian field', ax=ax2)
            p3 = ax3.contourf(x, y, np.abs(computed - analytical) / np.abs(analytical), 100,
                              locator=ticker.LogLocator())
            fig.colorbar(p3, label='Relative difference', ax=ax3)
            plt.tight_layout()
            plt.savefig(f'test_laplacian_numpy_{case}_order{order}.png')
    numpy_plot(test_laplacian_numpy, "poly")
    numpy_plot(test_laplacian_numpy_exp, "exp")
