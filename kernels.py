# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import math

from numba import cuda

# GRID -----------------------------------------------------------------------------------------------------------------

# The distance between two neighboring grid cell centers in t-direction.
DT = 0.5

# The distance between two neighboring grid cell centers in x-direction.
DX = 1

# The distance between two neighboring grid cell centers in y-direction (must be DX, as pressure computation
# implemented only for this case).
DY = DX

# @formatter:off
NEGATE_S_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def negate_s(scalar, result):
    """
    Negate a scalar.

    Args:
        scalar: The considered scalar.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = -scalar[x, y]


# @formatter:off
NEGATE_V_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def negate_v(vector, result):
    """
    Negate a vector.

    Args:
        vector: The considered vector.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = -vector[x, y, 0]
        result[x, y, 1] = -vector[x, y, 1]


# @formatter:off
SCALE_S_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def scale_s(scalar, factor, result):
    """
    Scale via
      result = factor * scalar

    Args:
        scalar: The considered scalar.
        factor: The scale factor.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = factor * scalar[x, y]


# @formatter:off
SCALE_V_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def scale_v(vector, factor, result):
    """
    Scale via
      result = factor * vector

    Args:
        vector: The considered vector.
        factor: The scale factor.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = factor * vector[x, y, 0]
        result[x, y, 1] = factor * vector[x, y, 1]


# @formatter:off
ADD_S_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def add_s(scalar_0, scalar_1, result):
    """
    Add
      result = scalar_0 + scalar_1

    Args:
        scalar_0: The considered first scalar.
        scalar_1: The considered second scalar.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = scalar_0[x, y] + scalar_1[x, y]


# @formatter:off
ADD_V_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def add_v(vector_0, vector_1, result):
    """
    Add
      result = vector_0 + vector_1

    Args:
        vector_0: The considered first vector.
        vector_1: The considered second vector.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = vector_0[x, y, 0] + vector_1[x, y, 0]
        result[x, y, 1] = vector_0[x, y, 1] + vector_1[x, y, 1]


# @formatter:off
ADD_CONSTANT_VECTOR_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def add_constant_vector(vector, constant_vector, result):
    """
    Add
      result = vector_0 + vector_1

    Args:
        vector: The considered vector.
        constant_vector: The considered constant vector.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = vector[x, y, 0] + constant_vector[0]
        result[x, y, 1] = vector[x, y, 1] + constant_vector[1]


# @formatter:off
MULTIPLY_SS_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def multiply_ss(scalar_0, scalar_1, result):
    """
    Add
      result = scalar_0 * scalar_1

    Args:
        scalar_0: The considered first scalar.
        scalar_1: The considered second scalar.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = scalar_0[x, y] * scalar_1[x, y]


# @formatter:off
MULTIPLY_SV_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def multiply_sv(scalar, vector, result):
    """
    Multiply
      result = scalar * vector

    Args:
        scalar: The considered scalar.
        vector: The considered vector.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = scalar[x, y] * vector[x, y, 0]
        result[x, y, 1] = scalar[x, y] * vector[x, y, 1]


# @formatter:off
DIVIDE_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def divide(vector, scalar, result):
    """
    Divice
      result = vector / scalar

    Args:
        vector: The considered vector.
        scalar: The considered scalar.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = vector[x, y, 0] / scalar[x, y]
        result[x, y, 1] = vector[x, y, 1] / scalar[x, y]


# @formatter:off
SCALE_ADD_S_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def scale_add_s(scalar_0, scalar_1, factor, result):
    """
    Scale and add via
      result = factor * scalar_0 + scalar_1

    Args:
        scalar_0: The considered first scalar.
        scalar_1: The considered second scalar.
        factor: The scale factor.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = factor * scalar_0[x, y] + scalar_1[x, y]


# @formatter:off
SCALE_ADD_V_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def scale_add_v(vector_0, vector_1, factor, result):
    """
    Scale and add via
      result = factor * vector_0 + vector_1

    Args:
        vector_0: The considered first vector.
        vector_1: The considered second vector.
        factor: The scale factor.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = factor * vector_0[x, y, 0] + vector_1[x, y, 0]
        result[x, y, 1] = factor * vector_0[x, y, 1] + vector_1[x, y, 1]


# @formatter:off
WEIGHTED_SUM_S_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def weighted_sum_s(scalar_0, scalar_1, scalar_2, scalar_3, factor_0, factor_1, factor_2, factor_3, result):
    """
    Compute a weighted sum via
      result = factor_0 * scalar_0 + ... + factor_3 * scalar_3

    Args:
        scalar_0: The considered first scalar.
        ...
        scalar_3: The considered last scalar.
        factor_0: The first scale factor.
        ...
        factor_3: The last scale factor.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = factor_0 * scalar_0[x, y] + factor_1 * scalar_1[x, y] + factor_2 * scalar_2[x, y] + \
                       factor_3 * scalar_3[x, y]


# @formatter:off
WEIGHTED_SUM_V_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def weighted_sum_v(vector_0, vector_1, vector_2, vector_3, factor_0, factor_1, factor_2, factor_3, result):
    """
    Compute a weighted sum via
      result = factor_0 * vector_0 + ... + factor_3 * vector_3

    Args:
        vector_0: The considered first vector.
        ...
        vector_3: The considered last vector.
        factor_0: The first scale factor.
        ...
        factor_3: The last scale factor.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = factor_0 * vector_0[x, y, 0] + factor_1 * vector_1[x, y, 0] + factor_2 * vector_2[x, y, 0] + \
                          factor_3 * vector_3[x, y, 0]
        result[x, y, 1] = factor_0 * vector_0[x, y, 1] + factor_1 * vector_1[x, y, 1] + factor_2 * vector_2[x, y, 1] + \
                          factor_3 * vector_3[x, y, 1]


# @formatter:off
SLICE_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def slice(vector, index, result):
    """
    Slice a vector according to (using such a slicing without a kernel directly on CUDADispatcher is very slow)
      result = vector[:, :, index]

    Args:
        vector: The considered vector.
        index: The slicing index.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = vector[x, y, index]


# @formatter:off
D_DX_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def d_dx(scalar, result):
    """
    Compute dscalar/dx via finite differences.

    Args:
        scalar: The considered scalar.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        if x == 0:
            result[x, y] = (scalar[x + 1, y] - scalar[x, y]) / DX
        elif x < scalar.shape[0] - 1:
            result[x, y] = (scalar[x + 1, y] - scalar[x - 1, y]) / (2 * DX)
        else:
            result[x, y] = (scalar[x, y] - scalar[x - 1, y]) / DX


# @formatter:off
D_DY_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def d_dy(scalar, result):
    """
    Compute dscalar/dy via finite differences.

    Args:
        scalar: The considered scalar.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        if y == 0:
            result[x, y] = (scalar[x, y + 1] - scalar[x, y]) / DY
        elif y < scalar.shape[1] - 1:
            result[x, y] = (scalar[x, y + 1] - scalar[x, y - 1]) / (2 * DY)
        else:
            result[x, y] = (scalar[x, y] - scalar[x, y - 1]) / DY


# @formatter:off
COMBINE_SHAPE_PER_CELL = (2,)
# @formatter:on
@cuda.jit
def combine(scalar_0, scalar_1, result):
    """
    Combine two scalars to a vector.

    Args:
        scalar_0: The first scalar.
        scalar_1: The second scalar.
        result: The result of the operator as a vector.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y, 0] = scalar_0[x, y]
        result[x, y, 1] = scalar_1[x, y]


# @formatter:off
FORWARD_EULER_S_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def forward_euler_s(scalar, dscalar_dt):
    """
    Compute the forward Euler update
      scalar <- scalar + DT*dscalar_dt

    Args:
        scalar: The considered scalar.
        dscalar_dt: The temporal derivative of the scalar.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        scalar[x, y] += DT * dscalar_dt[x, y]


# @formatter:off
FORWARD_EULER_V_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def forward_euler_v(vector, dvector_dt):
    """
    Compute the forward Euler update
      vector <- vector + DT*dvector_dt

    Args:
        vector: The considered vector.
        dvector_dt: The temporal derivative of the vector.
    """
    x, y = cuda.grid(2)
    if (x < vector.shape[0]) and y < (vector.shape[1]):
        vector[x, y, 0] += DT * dvector_dt[x, y, 0]
        vector[x, y, 1] += DT * dvector_dt[x, y, 1]


# @formatter:off
CLAMP_S_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def clamp_s(scalar, minimum, maximum):
    """
    Clamp a scalar within (min, max).

    Args:
        scalar: The considered scalar.
        minimum: The minimal value possible.
        maximum: The maximal value possible.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        if scalar[x, y] < minimum:
            scalar[x, y] = minimum
        elif scalar[x, y] > maximum:
            scalar[x, y] = maximum


# @formatter:off
CLAMP_V_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def clamp_v(vector, maximum_length):
    """
    Clamp the length of a vector such that it has maximally a given length.

    Args:
        vector: The considered vector.
        maximum_length: The maximally allowed length.
    """
    x, y = cuda.grid(2)
    if (x < vector.shape[0]) and y < (vector.shape[1]):
        length = (vector[x, y, 0] ** 2 + vector[x, y, 1] ** 2) ** 0.5
        if length > maximum_length:
            factor = maximum_length / length
            vector[x, y, 0] *= factor
            vector[x, y, 1] *= factor


# @formatter:off
LENGTH_SHAPE_PER_CELL = ()
# @formatter:on
@cuda.jit
def length(vector, result):
    """
    Compute the length of a vector.

    Args:
        vector: The considered vector.
        result: The result of the operator as a scalar.
    """
    x, y = cuda.grid(2)
    if (x < result.shape[0]) and y < (result.shape[1]):
        result[x, y] = (vector[x, y, 0] ** 2 + vector[x, y, 1] ** 2) ** 0.5


# @formatter:off
FIX_S_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def fix_s(scalar, center, radius, value):
    """
    Fix a scalar to a value within a given circle.

    Args:
        scalar: The considered scalar.
        center: The center of the circle.
        radius: The radius of the circle.
        value: The value to be set.
    """
    x, y = cuda.grid(2)
    if (x < scalar.shape[0]) and y < (scalar.shape[1]):
        dx = x - center[0]
        dy = y - center[1]
        if dx ** 2 + dy ** 2 < radius ** 2:
            scalar[x, y] = value


# @formatter:off
FIX_V_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def fix_v(vector, center, radius, value):
    """
    Fix a vector to a value within a given circle.

    Args:
        vector: The considered vector.
        center: The center of the circle.
        radius: The radius of the circle.
        value: The value to be set.
    """
    x, y = cuda.grid(2)
    if (x < vector.shape[0]) and y < (vector.shape[1]):
        dx = x - center[0]
        dy = y - center[1]
        if dx ** 2 + dy ** 2 < radius ** 2:
            vector[x, y, 0] = value[0]
            vector[x, y, 1] = value[1]


# @formatter:off
VERTICAL_EXPONENTIAL_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def vertical_exponential(scalar, minimum, maximum, only_boundary):
    """
    Set a scalar such that it rises vertically as
      scalar = s e^(-h)
    where h is the height and minimum and maximum are the values at the bottom and top.

    Args:
        scalar: The considered scalar.
        minimum: The minimum value.
        maximum: The maximum value.
        only_boundary: Set the scalar only at the boundary?
    """
    # We have
    #   scalar = a e^(-b*h)
    # with
    #   h = - y
    # such that
    #   scalar = a e^(b*y)
    # Hence,
    #   scalar = maximum e^(b*y)
    # because y=0 at the top. Hence,
    #   minimum = maximum e^(b*(scalar.shape[1]-1))
    # or
    #   b = log(minimum/maximum)/(scalar.shape[1]-1)
    x, y = cuda.grid(2)
    if (not only_boundary and (x < scalar.shape[0]) and y < (scalar.shape[1])) or \
            (only_boundary and ((x == 0) or (x == scalar.shape[0] - 1) or (y == 0) or (y == scalar.shape[1] - 1))):
        b = math.log(minimum / maximum) / (scalar.shape[1] - 1)
        scalar[x, y] = maximum * math.exp(b * y)


# @formatter:off
GRID_BOUNDARY_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def grid_boundary(vector, value):
    """
    Fix a vector on the grid boundary to a given value.

    Args:
        vector: The considered vector.
        value: The boundary value.
    """
    x, y = cuda.grid(2)
    if (x == 0) or (x == vector.shape[0] - 1) or (y == 0) or (y == vector.shape[1] - 1):
        vector[x, y, 0] = value[0]
        vector[x, y, 1] = value[1]


# @formatter:off
COMPUTE_FORCE_SHAPE_PER_CELL = None
# @formatter:on
@cuda.jit
def compute_force(rho, p, u, center, radius, force):
    """
    Compute the force applied on a circle.

    This computation is done heuristically on grid cell centers which have at least one neighboring wing cell and the
    used area element is assumed to be unity.

    Args:
        rho: The density.
        p: The pressure.
        u: The velocity.
        center: The center of the circle.
        radius: The radius of the circle.
        force: The computed force
    """
    x, y = cuda.grid(2)
    if (x < p.shape[0]) and y < (p.shape[1]):
        dx = x - center[0]
        dy = y - center[1]
        length2 = dx ** 2 + dy ** 2
        if length2 >= radius ** 2:
            one_cell_inside = False
            for xp in range(max(0, x - 1), min(x + 2, p.shape[0])):
                for yp in range(max(0, y - 1), min(y + 2, p.shape[1])):
                    dxp = xp - center[0]
                    dyp = yp - center[1]
                    if dxp ** 2 + dyp ** 2 < radius ** 2:
                        one_cell_inside = True
                        break
                if one_cell_inside:
                    break
            if one_cell_inside:
                factor = 1 / length2 ** 0.5
                normal = (factor * dx, factor * dy)
                pressure_force = (-p[x, y] * normal[0], -p[x, y] * normal[1])

                u_dot_normal = u[x, y, 0] * normal[0] + u[x, y, 1] * normal[1]
                momentum_flux = (rho[x, y] * u_dot_normal * normal[0], rho[x, y] * u_dot_normal * normal[1])

                force_per_cell = (pressure_force[0] + momentum_flux[0], pressure_force[1] + momentum_flux[1])

                for index in range(2):
                    cuda.atomic.add(force, index, force_per_cell[index])
