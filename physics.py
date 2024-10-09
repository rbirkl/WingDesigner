# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import numpy as np

from kernels import *

from maths import div, execute, get, grad, grad_div, laplacian_s, laplacian_v
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from typing import Tuple
from visualization import RESOLUTION

# CONSTANTS ------------------------------------------------------------------------------------------------------------

# The dynamic viscosity.
MU = 0.25

# The bulk viscosity.
ZETA = 0.25

# The specific gas constant R.
R = 1

# The temperature T.
T = 1

# The gravitational force in (x,y)-direction.
GRAVITY = (0, .005)

# The artificial diffusion coefficient D.
D = 1

# INITIAL CONFIGURATION ------------------------------------------------------------------------------------------------

# The initial density at the bottom and top of the computational domain.
INITIAL_RHO = (1, 0.3)

# The initial velocity in (x,y)-direction.
INITIAL_VELOCITY = (1, 0)

# The density of the wing.
WING_RHO = 0.7 * INITIAL_RHO[0]

# The radius of the wing.
WING_RADIUS = 10

# CONSTRAINTS ----------------------------------------------------------------------------------------------------------

# The allowed density range (boundaries inclusive).
RANGE_RHO = (0.01, 1)

# The maximally allowed speed.
MAX_SPEED = 1


def initialize_physics() -> Tuple[DeviceNDArray, DeviceNDArray]:
    """
    Initialize the physical fields.

    Returns:
        The density rho and the velocity u.
    """
    rho = cuda.device_array(RESOLUTION, dtype=np.float32)

    execute(vertical_exponential, rho, *INITIAL_RHO, False)
    execute(fix_s, rho, (RESOLUTION[0] / 2, RESOLUTION[1] / 2), WING_RADIUS, WING_RHO)

    u = cuda.device_array(rho.shape + (2,), dtype=rho.dtype)

    u[:, :, 0] = INITIAL_VELOCITY[0]
    u[:, :, 1] = INITIAL_VELOCITY[1]

    return cuda.to_device(rho), cuda.to_device(u)


def compute_p(rho: DeviceNDArray) -> DeviceNDArray:
    """
    Compute the pressure according to
        p = rho R T

    Args:
        rho: The density.

    Returns:
        The pressure p.
    """
    return execute(scale_s, rho, R * T)


def compute_drho_dt(rho: DeviceNDArray, u: DeviceNDArray) -> DeviceNDArray:
    """
    Compute
        drho/dt = - div (rho * u)
                  + D * max(DX,DY) * |u| * Laplacian rho
    where the last term is an artificial diffusion with diffusion coefficient D.

    Args:
        rho: The density.
        u: The velocity.

    Returns:
        The scalar drho/dt.
    """
    rho_u = execute(multiply_sv, rho, u)
    div_rho_u = div(rho_u)
    negative_div_rho_u = execute(negate_s, div_rho_u)

    factor = D * max(DX, DY)
    length_u = execute(length, u)
    laplacian = laplacian_s(rho)
    product = execute(multiply_ss, length_u, laplacian)

    return execute(scale_add_s, product, negative_div_rho_u, factor)


def compute_minus_ua_du_da(u: DeviceNDArray) -> DeviceNDArray:
    """
    Compute - (u * grad)u

    Args:
        u: The velocity.

    Returns:
        The computed field.
    """
    ux = get(u, 0)
    uy = get(u, 1)

    dux_dx = execute(d_dx, ux)
    dux_dy = execute(d_dy, ux)
    duy_dx = execute(d_dx, uy)
    duy_dy = execute(d_dy, uy)

    ux_dux_dx = execute(multiply_ss, ux, dux_dx)
    uy_dux_dy = execute(multiply_ss, uy, dux_dy)
    ux_duy_dx = execute(multiply_ss, ux, duy_dx)
    uy_duy_dy = execute(multiply_ss, uy, duy_dy)

    ua_dux_da = execute(add_s, ux_dux_dx, uy_dux_dy)
    ua_duy_da = execute(add_s, ux_duy_dx, uy_duy_dy)

    ua_du_da = execute(combine, ua_dux_da, ua_duy_da)

    return execute(negate_v, ua_du_da)


def compute_du_dt(u: DeviceNDArray, rho: DeviceNDArray) -> DeviceNDArray:
    """
    Compute
        du/dt   = - (u * grad)u - (grad p - mu * ( Delta u + div (grad u)^T) + ((2/3) mu - zeta) * grad div u) / rho
                  + D * max(DX,DY) * |u| * Laplacian u
    where the last term is an artificial diffusion with diffusion coefficient D.

    Args:
        u: The velocity.
        rho: The density.

    Returns:
        The vector du/dt.
    """
    minus_ua_du_da = compute_minus_ua_du_da(u)

    p = compute_p(rho)
    grad_p = grad(p)

    delta_u = laplacian_v(u)
    term_2 = execute(scale_v, delta_u, -MU)

    grad_div_u = grad_div(u)
    term_3 = execute(scale_v, grad_div_u, -(MU / 3 + ZETA))

    sum = execute(add_v, grad_p, term_2)
    sum = execute(add_v, sum, term_3)

    expression = execute(divide, sum, rho)
    expression = execute(negate_v, expression)

    du_dt = execute(add_v, minus_ua_du_da, expression)
    du_dt = execute(add_constant_vector, du_dt, GRAVITY)

    factor = D * max(DX, DY)
    length_u = execute(length, u)
    laplacian = laplacian_v(u)
    product = execute(multiply_sv, length_u, laplacian)

    return execute(scale_add_v, product, du_dt, factor)


def constrain_rho(rho: DeviceNDArray):
    """
    Constrain the density rho.

    Args:
        rho: The density.
    """
    execute(clamp_s, rho, *RANGE_RHO)

    execute(vertical_exponential, rho, *INITIAL_RHO, True)
    execute(fix_s, rho, (RESOLUTION[0] / 2, RESOLUTION[1] / 2), WING_RADIUS, WING_RHO)


def constrain_u(u: DeviceNDArray):
    """
    Constrain the velocity u.

    Args:
        u: The velocity.
    """
    execute(clamp_v, u, MAX_SPEED)

    execute(grid_boundary, u, INITIAL_VELOCITY)
    execute(fix_v, u, (RESOLUTION[0] / 2, RESOLUTION[1] / 2), WING_RADIUS, (0, 0))


def update(rho: DeviceNDArray, u: DeviceNDArray):
    """
    Update the density rho and velocity u via 4th order Runge-Kutta according to
        drho/dt = - div (rho * u)
        du/dt   = - (u * grad)u - (grad p - mu * ( Delta u + div (grad u)^T) + ((2/3) mu - zeta) * grad div u) / rho

    Args:
        rho: The density.
        u: The velocity.
    """
    k1_rho = compute_drho_dt(rho, u)
    k1_u = compute_du_dt(u, rho)

    rho_prime = execute(scale_add_s, k1_rho, rho, DT / 2)
    u_prime = execute(scale_add_v, k1_u, u, DT / 2)

    constrain_rho(rho_prime)
    constrain_u(u_prime)

    k2_rho = compute_drho_dt(rho_prime, u_prime)
    k2_u = compute_du_dt(u_prime, rho_prime)

    rho_prime = execute(scale_add_s, k2_rho, rho, DT / 2)
    u_prime = execute(scale_add_v, k2_u, u, DT / 2)

    constrain_rho(rho_prime)
    constrain_u(u_prime)

    k3_rho = compute_drho_dt(rho_prime, u_prime)
    k3_u = compute_du_dt(u_prime, rho_prime)

    rho_prime = execute(scale_add_s, k3_rho, rho, DT)
    u_prime = execute(scale_add_v, k3_u, u, DT)

    constrain_rho(rho_prime)
    constrain_u(u_prime)

    k4_rho = compute_drho_dt(rho_prime, u_prime)
    k4_u = compute_du_dt(u_prime, rho_prime)

    drho_dt = execute(weighted_sum_s, k1_rho, k2_rho, k3_rho, k4_rho, 1 / 6, 1 / 3, 1 / 3, 1 / 6)
    du_dt = execute(weighted_sum_v, k1_u, k2_u, k3_u, k4_u, 1 / 6, 1 / 3, 1 / 3, 1 / 6)

    execute(forward_euler_s, rho, drho_dt)
    execute(forward_euler_v, u, du_dt)

    constrain_rho(rho)
    constrain_u(u)
