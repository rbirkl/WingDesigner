# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import numpy as np

from src.kernels import *

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher
from numpy import ndarray
from scipy.special import comb
from src.visualization import RESOLUTION
from typing import Optional, Tuple

# CUDA -----------------------------------------------------------------------------------------------------------------

# The number of threads per block in x- and y-direction.
THREADS_PER_BLOCK = (32, 32)


def execute(kernel: CUDADispatcher, *args: Tuple[DeviceNDArray, ...], result: Optional[DeviceNDArray] = None) -> \
        Optional[DeviceNDArray]:
    """
    Executes a CUDA kernel on the provided fields.

    Args:
        kernel: The CUDA kernel to execute of signature (*fields, result):
        *args: The args to be processed by the kernel, where the first one has to be a field.
        result: The field where the results are written to

    Result:
        The resulting field.
    """
    blocks_per_grid_x = (RESOLUTION[0] + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
    blocks_per_grid_y = (RESOLUTION[1] + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    arguments = args

    if result is None:
        shape_per_cell = eval(kernel.__name__.upper() + "_SHAPE_PER_CELL")
        if shape_per_cell is not None:
            result = cuda.device_array(args[0].shape[:2] + shape_per_cell, dtype=np.float32)
    if result is not None:
        arguments += (result,)

    kernel[blocks_per_grid, THREADS_PER_BLOCK](*arguments)

    if result is not None:
        return result


def get(vector: DeviceNDArray, index: int) -> DeviceNDArray:
    """
    Get a component of a vector.

    Args:
        vector: The considered vector.
        index: The index of the component.

    Returns:
        The component.
    """
    return execute(slice, vector, index)


def grad(scalar: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the gradient of a scalar.

    Args:
        scalar: The considered scalar.

    Returns:
        The gradient.
    """
    dscalar_dx = execute(d_dx, scalar)
    dscalar_dy = execute(d_dy, scalar)

    return execute(combine, dscalar_dx, dscalar_dy)


def div(vector: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the divergence of a vector.

    Args:
        vector: The considered vector.

    Returns:
        The divergence.
    """
    ux = get(vector, 0)
    uy = get(vector, 1)

    dvector_x_dx = execute(d_dx, ux)
    dvector_y_dy = execute(d_dy, uy)

    return execute(add_s, dvector_x_dx, dvector_y_dy)


def laplacian_s(scalar: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the Laplacian of a scalar.

    Args:
        scalar: The considered scalar.

    Returns:
        The Laplacian.
    """
    dscalar_dx = execute(d_dx, scalar)
    dscalar_dy = execute(d_dy, scalar)

    d2scalar_dx2 = execute(d_dx, dscalar_dx)
    d2scalar_dy2 = execute(d_dy, dscalar_dy)

    return execute(add_s, d2scalar_dx2, d2scalar_dy2)


def laplacian_v(vector: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the Laplacian of a vector.

    Args:
        vector: The considered vector.

    Returns:
        The Laplacian.
    """
    ux = get(vector, 0)
    uy = get(vector, 1)

    dvector_x_dx = execute(d_dx, ux)
    dvector_x_dy = execute(d_dy, ux)
    dvector_y_dx = execute(d_dx, uy)
    dvector_y_dy = execute(d_dy, uy)

    d2vector_x_dx2 = execute(d_dx, dvector_x_dx)
    d2vector_x_dy2 = execute(d_dy, dvector_x_dy)
    d2vector_y_dx2 = execute(d_dx, dvector_y_dx)
    d2vector_y_dy2 = execute(d_dy, dvector_y_dy)

    laplacian_vector_x = execute(add_s, d2vector_x_dx2, d2vector_x_dy2)
    laplacian_vector_y = execute(add_s, d2vector_y_dx2, d2vector_y_dy2)

    return execute(combine, laplacian_vector_x, laplacian_vector_y)


def grad_div(vector: DeviceNDArray) -> DeviceNDArray:
    """
    Compute div (grad vector)^T.

    Args:
        vector: The considered vector.

    Returns:
        The computed expression.
    """
    div_vector = div(vector)

    ddiv_vector_dx = execute(d_dx, div_vector)
    ddiv_vector_dy = execute(d_dy, div_vector)

    return execute(combine, ddiv_vector_dx, ddiv_vector_dy)


def bezier_curve(control_points: ndarray, number_points: int) -> ndarray:
    """
    Compute a Bezier curve.

    Args:
        control_points: The paramters of the curve.
        number_points: The number of points of the output.

    Returns:
        The computed Bezier polygon.
    """
    control_points = np.array(control_points)

    n = len(control_points) - 1
    t_values = np.linspace(0, 1, number_points)

    curve_points = np.zeros((number_points, 2))
    for i in range(number_points):
        t = t_values[i]
        for j in range(n + 1):
            curve_points[i] += comb(n, j) * (t ** j) * ((1 - t) ** (n - j)) * control_points[j]

    return np.vstack([curve_points, curve_points[0]], dtype=np.float32)
