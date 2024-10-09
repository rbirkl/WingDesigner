# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import pygame
import pygame.gfxdraw

import numpy as np

from numba.cuda.dispatcher import CUDADispatcher
from pygame.surface import Surface

# The resolution as (width, height).
RESOLUTION = (256, 128)

# The stride which determines how many vectors are visualized.
STRIDE = 20

# The scale of the vectors when visualized.
SCALE = 1000

# The size of the visualized arrow heads.
HEAD_SIZE = 5

# The angle of the visualized arrows in terms of 2*pi.
HEAD_ANGLE = 0.1

# The arrow color in (red, green, blue).
COLOR_ARROW = (0, 0, 255)


def initialize_visualization() -> Surface:
    """
    Initialize the visualization.

    Returns:
        The visualization screen.
    """
    pygame.init()
    return pygame.display.set_mode(RESOLUTION)


def visualize_scalar(screen: Surface, scalar: CUDADispatcher):
    """
    Visualizes a scalar field.

    Args:
        screen: The pygame screen.
        scalar: The scalar of shape (width, height).
    """
    scalar = scalar.copy_to_host()
    field = ((scalar+1) * 127.5).astype(np.uint8)
    field_rgb = np.stack((field, field, field), axis=-1)
    surface = pygame.surfarray.make_surface(field_rgb)
    screen.blit(surface, (0, 0))


def visualize_vector(screen: Surface, vector: CUDADispatcher):
    """
    Visualizes a vector field.

    Args:
        screen: The pygame screen.
        vector: The vector of shape (width, height, 2).
    """
    width = screen.get_width()
    height = screen.get_height()

    vector = vector.copy_to_host()

    for x in range(0, width, STRIDE):
        for y in range(0, height, STRIDE):
            tail = (x, y)
            head = (x + SCALE * vector[x, y, 0], y + SCALE * vector[x, y, 1])

            int_head = [int(coord) for coord in head]
            pygame.gfxdraw.aatrigon(screen, *tail, *int_head, *int_head, COLOR_ARROW)

            angle = np.arctan2(head[1] - tail[1], head[0] - tail[0])
            angle_minus = angle - HEAD_ANGLE*np.pi
            angle_plus = angle + HEAD_ANGLE*np.pi
            arrow_points = [(head[0] - HEAD_SIZE * np.cos(angle_minus), head[1] - HEAD_SIZE * np.sin(angle_minus)),
                            (head[0] - HEAD_SIZE * np.cos(angle_plus), head[1] - HEAD_SIZE * np.sin(angle_plus)),
                            head]

            pygame.gfxdraw.filled_polygon(screen, arrow_points, COLOR_ARROW)
            pygame.gfxdraw.aapolygon(screen, arrow_points, COLOR_ARROW)
