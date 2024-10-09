# Copyright (c) 2024 Reiner Birkl

import numpy as np

from kernels import *

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher
from typing import Tuple
from visualization import RESOLUTION

# The viscosity.
NU = 0.001

# The number of threads per block in x- and y-direction.
THREADS_PER_BLOCK = (16, 16)

# The standard deviation of the initial Gaussian stream function psi.
SIGMA = 0.3


def execute_kernel(kernel: CUDADispatcher, *fields: Tuple[DeviceNDArray, ...]):
    """
    Executes a CUDA kernel on the provided fields.

    Args:
        kernel: The CUDA kernel to execute.
        *fields: The fields to be processed by the kernel.
    """
    blocks_per_grid_x = (RESOLUTION[0] + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
    blocks_per_grid_y = (RESOLUTION[1] + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    kernel[blocks_per_grid, THREADS_PER_BLOCK](*fields)


def initialize_physics() -> DeviceNDArray:
    """
    Initialize the stream function psi.

    Returns:
        The stream function psi.
    """
    x = np.linspace(-1, 1, RESOLUTION[0])
    y = np.linspace(-1, 1, RESOLUTION[1])
    y, x = np.meshgrid(y, x)

    psi = np.exp(-0.5 * (((x + 0.5) / SIGMA) ** 2 + (y / SIGMA) ** 2)) \
          - np.exp(-0.5 * (((x - 0.5) / SIGMA) ** 2 + (y / SIGMA) ** 2))


    return cuda.to_device(psi)


def compute_gradient_components(scalar: DeviceNDArray) -> Tuple[DeviceNDArray, DeviceNDArray]:
    """
    Compute the gradient components of a scalar.

    Args:
       scalar: The scalar considered.

    Returns:
        The gradient components dscalar/dx, dscalar/dy.
    """
    dscalar_dx = cuda.device_array_like(scalar)
    dscalar_dy = cuda.device_array_like(scalar)

    execute_kernel(d_dx, scalar, dscalar_dx)
    execute_kernel(d_dy, scalar, dscalar_dy)

    return dscalar_dx, dscalar_dy


def compute_laplacian(scalar: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the Laplacian of a scalar.

    Args:
        scalar: The scalar considered.

    Returns:
        The Laplacian Delta scalar.
    """
    dscalar_dx, dscalar_dy = compute_gradient_components(scalar)

    d2scalar_dx2 = cuda.device_array_like(scalar)
    d2scalar_dy2 = cuda.device_array_like(scalar)

    execute_kernel(d_dx, dscalar_dx, d2scalar_dx2)
    execute_kernel(d_dy, dscalar_dy, d2scalar_dy2)

    laplacian = cuda.device_array_like(scalar)
    execute_kernel(add, d2scalar_dx2, d2scalar_dy2, laplacian)

    return laplacian


def compute_velocity(psi: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the velocity u from the stream function psi according to
      u = (dpsi/dy, -dpsi/dx)

    Args:
       psi: The stream function psi.

    Returns:
        The velocity.
    """
    dpsi_dx, dpsi_dy = compute_gradient_components(psi)

    negated_dpsi_dx = dpsi_dx
    execute_kernel(negate, negated_dpsi_dx)

    velocity = cuda.device_array(psi.shape + (2,), dtype=psi.dtype)

    execute_kernel(combine_arrays, dpsi_dy, negated_dpsi_dx, velocity)

    return velocity


def compute_dpsi_dt(psi: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the temporal derivative of the stream function psi according to
      dpsi(x,y)/dt = (1/(2*pi)) int dx'dy' ln ((x-x')^2 + (y-y')^2)^(1/2) (  dpsi(x',y')/dx' * Delta' dpsi(x',y')/dy'
                                                                           - dpsi(x',y')/dy' * Delta' dpsi(x',y')/dx'
                                                                           + nu * Delta'^2 psi(x',y') )

    Args:
        psi: The stream function psi.

    Returns:
        The derivative dpsi/dt.
    """
    dpsi_dx, dpsi_dy = compute_gradient_components(psi)

    delta_dpsi_dx = compute_laplacian(dpsi_dx)
    delta_dpsi_dy = compute_laplacian(dpsi_dy)

    delta_psi = compute_laplacian(psi)
    delta2_psi = compute_laplacian(delta_psi)

    result = cuda.device_array_like(psi)

    execute_kernel(dpsi_dt, dpsi_dx, dpsi_dy, delta_dpsi_dx, delta_dpsi_dy, delta2_psi, NU, result)

    return result


def update_psi(psi: DeviceNDArray):
    """
    Update the stream function psi via forward Euler according to
        psi(t+dt)=psi(t+dt)+dt*dpsi/dt

    Args:
        psi: The stream function psi.
    """
    dpsi_dt = compute_dpsi_dt(psi)
    execute_kernel(forward_euler, psi, dpsi_dt)
