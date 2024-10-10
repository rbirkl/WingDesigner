# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import pygame
import warnings

import numpy as np

from ai import COLOR_WING, generate_labels, optimize, train
from numba import NumbaPerformanceWarning
from physics import initialize_physics, update_physics, WING_CONTROL_POINTS
from pygame.surface import Surface
from torch import Tensor
from visualization import initialize_visualization, visualize_scalar, visualize_vector


def show_optimum(screen: Surface, optimized_input: Tensor):
    """
    Show the optimal solution.

    Args:
        screen: The visualization screen.
        optimized_input: The optimized input.
    """
    wing_control_points = []
    for index, wing_control_point in enumerate(WING_CONTROL_POINTS):
        point = wing_control_point
        if (index > 0) and (index < len(WING_CONTROL_POINTS) - 1):
            point = tuple(optimized_input[index - 1].numpy())
        wing_control_points.append(point)
    wing_control_points = tuple(wing_control_points)

    rho, u, wing_position, wing = initialize_physics(wing_control_points)

    frame = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE)):
                pygame.quit()
                exit()

        visualize_scalar(screen, rho)
        visualize_vector(screen, u)
        pygame.gfxdraw.aapolygon(screen, np.array(wing_position) + wing, COLOR_WING)

        wing_position, force = update_physics(rho, u, wing_position, wing)

        pygame.display.flip()

        frame += 1

        print(f"Force: ({force[0]:.1f}, {force[1]:.1f})")


def main():
    warnings.simplefilter('ignore', NumbaPerformanceWarning)

    screen = initialize_visualization("Training")

    inputs, labels = generate_labels(screen)
    pygame.quit()

    model = train(inputs, labels)

    optimized_input = optimize(inputs, model)

    screen = initialize_visualization("Optimum")

    show_optimum(screen, optimized_input)


if __name__ == "__main__":
    main()
