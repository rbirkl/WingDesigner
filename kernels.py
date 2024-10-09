# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import math

from numba import cuda

# The distance between two neighboring grid cell centers in t-direction.
DT = 0.05

# The distance between two neighboring grid cell centers in x-direction.
DX = 1

# The distance between two neighboring grid cell centers in y-direction.
DY = 1

# The stride to simplify the inversion of the Laplacian (the solution is slowest and most exact for STRIDE=1).
STRIDE=4


@cuda.jit
def negate(scalar):
    """
    Negate a scalar.

    Args:
        scalar: The scalar to be negated.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        scalar[x, y] = -scalar[x, y]


@cuda.jit
def add(scalar_0, scalar_1, sum):
    """
    Add two scalars.

    Args:
        scalar_0: The first scalar.
        scalar_1: The second scalar.
        sum: The sum of the scalars.
    """
    x, y = cuda.grid(2)
    if (x < sum.shape[0]) and y < (sum.shape[1]):
        sum[x, y] = scalar_0[x, y] + scalar_1[x, y]


@cuda.jit
def d_dx(scalar, dscalar_dx):
    """
    Compute dscalar/dx via finite differences.

    Args:
        scalar: The considered scalar.
        dscalar_dx: The derivative dscalar/dx.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        if x == 0:
            dscalar_dx[x, y] = (scalar[x + 1, y] - scalar[x, y]) / DX
        elif x < scalar.shape[0] - 1:
            dscalar_dx[x, y] = (scalar[x + 1, y] - scalar[x - 1, y]) / (2 * DX)
        else:
            dscalar_dx[x, y] = (scalar[x, y] - scalar[x - 1, y]) / DX


@cuda.jit
def d_dy(scalar, dscalar_dy):
    """
    Compute dscalar/dy via finite differences.

    Args:
        scalar: The considered scalar.
        dscalar_dy: The derivative dscalar/dy.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        if y == 0:
            dscalar_dy[x, y] = (scalar[x, y + 1] - scalar[x, y]) / DY
        elif y < scalar.shape[1] - 1:
            dscalar_dy[x, y] = (scalar[x, y + 1] - scalar[x, y - 1]) / (2 * DY)
        else:
            dscalar_dy[x, y] = (scalar[x, y] - scalar[x, y - 1]) / DY


@cuda.jit
def combine_arrays(scalar_0, scalar_1, vector):
    """
    Combine two scalars to a vector.

    Args:
        scalar_0: The first scalar.
        scalar_1: The second scalar.
        vector: The vector to be determined.
    """
    x, y = cuda.grid(2)
    if (x < vector.shape[0]) and y < (vector.shape[1]):
        vector[x, y, 0] = scalar_0[x, y]
        vector[x, y, 1] = scalar_1[x, y]


@cuda.jit
def dpsi_dt(dpsi_dx, dpsi_dy, delta_dpsi_dx, delta_dpsi_dy, delta2_psi, nu, result):
    """
    Compute dpsi_dt according to
      result(x,y) = (1/(2*pi)) int dx'dy' ln ((x-x')^2 + (y-y')^2)^(1/2) (  dpsi_dx(x',y') * delta_dpsi_dy(x',y')
                                                                          - dpsi_dy(x',y') * delta_dpsi_dx(x',y')
                                                                          + nu * delta2_psi(x',y') )

    Args:
        dpsi_dx: The x-derivative of the stream function psi.
        dpsi_dy: The y-derivative of the stream function psi.
        delta_dpsi_dx: The Laplacian of the x-derivative of the stream function psi.
        delta_dpsi_dy: The Laplacian of the y-derivative of the stream function psi.
        delta2_psi: The Laplacian of the Laplacian of the stream function psi.
        nu: The viscosity.
        result: The result dpsi_dt.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        sum = 0

        for xp in range(0,result.shape[0],STRIDE):
            for yp in range(0,result.shape[1],STRIDE):
                difference_x = x - xp
                difference_y = y - yp

                distance = (difference_x ** 2 + difference_y ** 2) ** 0.5
                if distance !=0:
                    logarithm = math.log(distance)

                    integrand = logarithm * (
                            dpsi_dx[xp, yp] * delta_dpsi_dy[xp, yp] - dpsi_dy[xp, yp] * delta_dpsi_dx[xp, yp] +
                            nu * delta2_psi[xp, yp])
                    sum += integrand * DX * DY

        result[x, y] = sum *STRIDE**2/ (2 * math.pi)


@cuda.jit
def forward_euler(scalar, dscalar_dt):
    """
    Compute the forward Euler update
      scalar <- scalar + DT*dscalar_dt

    Args:
        scalar: The considered scalar.
        dscalar_dt: The temporal derivative of the scalar.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        scalar[x,y] = scalar[x, y] + DT * dscalar_dt[x, y]
