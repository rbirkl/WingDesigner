# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import pygame

import numpy as np

from physics import initialize_physics, update_physics
from time import time
from visualization import initialize_visualization, visualize_scalar, visualize_vector

# The wing color in (red, green, blue).
COLOR_WING = (255, 0, 0)


def main():
    screen = initialize_visualization()

    rho, u, wing_position, wing = initialize_physics()

    frame = 0
    start_time = time()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE)):
                pygame.quit()
                exit()

        visualize_scalar(screen, rho)
        visualize_vector(screen, u)
        pygame.gfxdraw.aapolygon(screen, np.array(wing_position) + wing, COLOR_WING)

        wing_position = update_physics(rho, u, wing_position, wing)

        pygame.display.flip()

        frame += 1
        end_time = time()
        delta_time = end_time - start_time
        print(f"Frame: {frame}, FPS: {frame / delta_time:.1f}")


if __name__ == "__main__":
    main()
