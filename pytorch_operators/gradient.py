########################################################################################################################
#                                                                                                                      #
#                                                 Gradient operators                                                   #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 02.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch


def gradient_scalar(field, dx, dy):
    """
    Calculates the gradient of a scalar field (second order accurate, decentered on the boundaries).

    Parameters
    ----------
    field : torch.Tensor
        Input 1D field: tensor of size (batch_size, 1, H, W)

    dx, dy : float
        Spatial step for W and H directions

    Returns
    -------
    torch.Tensor
        Output gradient field: tensor of size (batch_size, 2, H, W)
    """

    # Create gradient tensor with shape (batch_size, 2, h, w)
    gradient = torch.zeros(field.size(0), 2, field.size(2), field.size(3)).type(field.type())

    # Check sizes
    assert field.dim() == 4 and gradient.dim() == 4, 'Dimension mismatch'

    assert field.is_contiguous() and gradient.is_contiguous(), 'Input is not contiguous'

    # Compute gradient (correction using linear interpolation at boundaries)
    # i <-> y and j <-> x, so gradient[:, 0, :, :] is gradient_x and gradient[:, 1, :, :] is gradient_y
    # center of the array
    gradient[:, 0, :, 1:-1] = (field[:, 0, :, 2:] - field[:, 0, :, :-2]) / (2 * dx)
    gradient[:, 1, 1:-1, :] = (field[:, 0, 2:, :] - field[:, 0, :-2, :]) / (2 * dy)

    # array sides except corners (respectively upper, lower, left and right sides)
    gradient[:, 1, 0, 1:-1] = (4 * field[:, 0, 1, 1:-1] - 3 * field[:, 0, 0, 1:-1] - field[:, 0, 2, 1:-1]) / (2 * dy)
    gradient[:, 1, -1, 1:-1] = (3 * field[:, 0, -1, 1:-1] - 4 * field[:, 0, -2, 1:-1] + field[:, 0, -3, 1:-1]) / \
                               (2 * dy)
    gradient[:, 0, 1:-1, 0] = (4 * field[:, 0, 1:-1, 1] - 3 * field[:, 0, 1:-1, 0] - field[:, 0, 1:-1, 2]) / \
                              (2 * dx)
    gradient[:, 0, 1:-1, -1] = (3 * field[:, 0, 1:-1, -1] - 4 * field[:, 0, 1:-1, -2] + field[:, 0, 1:-1, -3]) / \
                               (2 * dx)

    # corners (respectively upper left, upper right, lower left and lower right)
    gradient[:, 0, 0, 0] = (4 * field[:, 0, 0, 1] - 3 * field[:, 0, 0, 0] - field[:, 0, 0, 2]) / (2 * dx)
    gradient[:, 1, 0, 0] = (4 * field[:, 0, 1, 0] - 3 * field[:, 0, 0, 0] - field[:, 0, 2, 0]) / (2 * dy)
    gradient[:, 0, -1, 0] = (4 * field[:, 0, -1, 1] - 3 * field[:, 0, -1, 0] - field[:, 0, -1, 2]) / (2 * dx)
    gradient[:, 1, -1, 0] = (3 * field[:, 0, -1, 0] - 4 * field[:, 0, -2, 0] + field[:, 0, -3, 0]) / (2 * dy)
    gradient[:, 0, 0, -1] = (3 * field[:, 0, 0, -1] - 4 * field[:, 0, 0, -2] + field[:, 0, 0, -3]) / (2 * dx)
    gradient[:, 1, 0, -1] = (4 * field[:, 0, 1, -1] - 3 * field[:, 0, 0, -1] - field[:, 0, 2, -1]) / (2 * dy)
    gradient[:, 0, -1, -1] = (3 * field[:, 0, -1, -1] - 4 * field[:, 0, -1, -2] + field[:, 0, -1, -3]) / (2 * dx)
    gradient[:, 1, -1, -1] = (3 * field[:, 0, -1, -1] - 4 * field[:, 0, -2, -1] + field[:, 0, -3, -1]) / (2 * dy)

    return gradient


def gradient_diag(field, dx, dy):
    """
    Calculates the diagonal terms of the gradient of a vector field (second order accurate, decentered
    on the boundaries).

    Parameters
    ----------
    field : torch.Tensor
        Input 2D field: tensor of size (batch_size, 2, H, W)

    dx, dy : float
        Spatial step for W and H directions

    Returns
    -------
    torch.Tensor
        Output diagonal gradient field: tensor of size (batch_size, 2, H, W)
    """

    # Create gradient tensor with shape (batch_size, 2, h, w)
    gradient = torch.zeros(field.size(0), 2, field.size(2), field.size(3)).type(field.type())

    # Check sizes
    assert field.dim() == 4 and gradient.dim() == 4, 'Dimension mismatch'
    assert field.size(1) == 2, 'field is not 2D'

    assert field.is_contiguous() and gradient.is_contiguous(), 'Input is not contiguous'

    # Compute gradient (correction using linear interpolation at boundaries)
    # i <-> y and j <-> x, so gradient[:, 0, :, :] is gradient_x and gradient[:, 1, :, :] is gradient_y
    # center of the array
    gradient[:, 0, 1:-1, 1:-1] = (field[:, 0, 1:-1, 2:] - field[:, 0, 1:-1, :-2]) / (2 * dx)
    gradient[:, 1, 1:-1, 1:-1] = (field[:, 1, 2:, 1:-1] - field[:, 1, :-2, 1:-1]) / (2 * dy)

    # array sides except corners (respectively upper, lower, left and right sides)
    gradient[:, 0, 0, 1:-1] = (field[:, 0, 0, 2:] - field[:, 0, 0, :-2]) / (2 * dx)
    gradient[:, 1, 0, 1:-1] = (4 * field[:, 1, 1, 1:-1] - 3 * field[:, 1, 0, 1:-1] - field[:, 1, 2, 1:-1]) / (2 * dy)
    gradient[:, 0, -1, 1:-1] = (field[:, 0, -1, 2:] - field[:, 0, -1, :-2]) / (2 * dx)
    gradient[:, 1, -1, 1:-1] = (3 * field[:, 1, -1, 1:-1] - 4 * field[:, 1, -2, 1:-1] + field[:, 1, -3, 1:-1]) / \
                               (2 * dy)
    gradient[:, 0, 1:-1, 0] = (4 * field[:, 0, 1:-1, 1] - 3 * field[:, 0, 1:-1, 0] - field[:, 0, 1:-1, 2]) / \
                              (2 * dx)
    gradient[:, 1, 1:-1, 0] = (field[:, 1, 2:, 0] - field[:, 1, :-2, 0]) / (2 * dy)
    gradient[:, 0, 1:-1, -1] = (3 * field[:, 0, 1:-1, -1] - 4 * field[:, 0, 1:-1, -2] + field[:, 0, 1:-1, -3]) / \
                               (2 * dx)
    gradient[:, 1, 1:-1, -1] = (field[:, 1, 2:, -1] - field[:, 1, :-2, -1]) / (2 * dy)

    # corners (respectively upper left, upper right, lower left and lower right)
    gradient[:, 0, 0, 0] = (4 * field[:, 0, 0, 1] - 3 * field[:, 0, 0, 0] - field[:, 0, 0, 2]) / (2 * dx)
    gradient[:, 1, 0, 0] = (4 * field[:, 0, 1, 0] - 3 * field[:, 0, 0, 0] - field[:, 0, 2, 0]) / (2 * dy)
    gradient[:, 0, -1, 0] = (4 * field[:, 0, -1, 1] - 3 * field[:, 0, -1, 0] - field[:, 0, -1, 2]) / (2 * dx)
    gradient[:, 1, -1, 0] = (3 * field[:, 1, -1, 0] - 4 * field[:, 1, -2, 0] + field[:, 1, -3, 0]) / (2 * dy)
    gradient[:, 0, 0, -1] = (3 * field[:, 0, 0, -1] - 4 * field[:, 0, 0, -2] + field[:, 0, 0, -3]) / (2 * dx)
    gradient[:, 1, 0, -1] = (4 * field[:, 1, 1, -1] - 3 * field[:, 1, 0, -1] - field[:, 1, 2, -1]) / (2 * dy)
    gradient[:, 0, -1, -1] = (3 * field[:, 0, -1, -1] - 4 * field[:, 0, -1, -2] + field[:, 0, -1, -3]) / (2 * dx)
    gradient[:, 1, -1, -1] = (3 * field[:, 1, -1, -1] - 4 * field[:, 1, -2, -1] + field[:, 1, -3, -1]) / (2 * dy)

    return gradient


def gradient_vector(field, dx, dy):
    """
    Calculates the gradient of a vector field (second order accurate, decentered on the boundaries).

    Parameters
    ----------
    field : torch.Tensor
        Input 2D field: tensor of size (batch_size, 2, H, W)

    dx, dy : float
        Spatial step for W and H directions

    Returns
    -------
    torch.Tensor
        Output gradient field: tensor of size (batch_size, 2, H, W)
    """

    # Create gradient tensor with shape (batch_size, h, w)
    gradient = torch.zeros_like(field).type(field.type())

    # Check sizes
    assert field.dim() == 4 and gradient.dim() == 4, 'Dimension mismatch'

    assert field.is_contiguous() and gradient.is_contiguous(), 'Input is not contiguous'

    # Compute gradient (correction using linear interpolation at boundaries)
    # i <-> y and j <-> x, so gradient[:, 0, :, :] is gradient_x and gradient[:, 1, :, :] is gradient_y
    # center of the array
    gradient[:, 1:-1, 1:-1] = (field[:, 2:, 1:-1] + field[:, :-2, 1:-1] - 2 * field[:, 1:-1, 1:-1]) / dy**2 + \
                              (field[:, 1:-1, 2:] + field[:, 1:-1, :-2] - 2 * field[:, 1:-1, 1:-1]) / dx**2

    # array sides except corners (respectively upper, lower, left and right sides)
    gradient[:, 0, 1:-1] = (2 * field[:, 0, 1:-1] - 5 * field[:, 1, 1:-1] + 4 * field[:, 2, 1:-1] - field[:, 3, 1:-1]) / dy**2 + \
                           (field[:, 0, 2:] + field[:, 0, :-2] - 2 * field[:, 0, 1:-1]) / dx**2
    gradient[:, -1, 1:-1] = (2 * field[:, -1, 1:-1] - 5 * field[:, -2, 1:-1] + 4 * field[:, -3, 1:-1] - field[:, -4, 1:-1]) / dy**2 + \
                            (field[:, -1, 2:] + field[:, -1, :-2] - 2 * field[:, -1, 1:-1]) / dx**2
    gradient[:, 1:-1, 0] = (field[:, 2:, 0] + field[:, :-2, 0] - 2 * field[:, 1:-1, 0]) / dy**2 + \
                           (2 * field[:, 1:-1, 0] - 5 * field[:, 1:-1, 1] + 4 * field[:, 1:-1, 2] - field[:, 1:-1, 3]) / dx**2
    gradient[:, 1:-1, -1] = (field[:, 2:, -1] + field[:, :-2, -1] - 2 * field[:, 1:-1, -1]) / dy**2 + \
                            (2 * field[:, 1:-1, -1] - 5 * field[:, 1:-1, -2] + 4 * field[:, 1:-1, -3] - field[:, 1:-1, -4]) / dx**2

    # corners (respectively upper left, upper right, lower left and lower right)
    gradient[:, 0, 0] = (2 * field[:, 0, 0] - 5 * field[:, 1, 0] + 4 * field[:, 2, 0] - field[:, 3, 0]) / dy**2 + \
                        (2 * field[:, 0, 0] - 5 * field[:, 0, 1] + 4 * field[:, 0, 2] - field[:, 0, 3]) / dx**2
    gradient[:, 0, -1] = (2 * field[:, 0, -1] - 5 * field[:, 1, -1] + 4 * field[:, 2, -1] - field[:, 3, -1]) / dy**2 + \
                         (2 * field[:, 0, -1] - 5 * field[:, 0, -2] + 4 * field[:, 0, -3] - field[:, 0, -4]) / dx**2
    gradient[:, -1, 0] = (2 * field[:, -1, 0] - 5 * field[:, -2, 0] + 4 * field[:, -3, 0] - field[:, -4, 0]) / dy**2 + \
                         (2 * field[:, -1, 0] - 5 * field[:, -1, 1] + 4 * field[:, -1, 2] - field[:, -1, 3]) / dx**2
    gradient[:, -1, -1] = (2 * field[:, -1, -1] - 5 * field[:, -2, -1] + 4 * field[:, -3, -1] - field[:, -4, -1]) / dy**2 + \
                          (2 * field[:, 0, -1] - 5 * field[:, 0, -2] + 4 * field[:, 0, -3] - field[:, 0, -4]) / dx**2

    return gradient
