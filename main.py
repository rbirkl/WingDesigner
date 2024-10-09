# Copyright (c) 2024 Reiner Birkl
#
# This file is part of the Wing Designer project.
#
# Wing Designer is licensed under license in the LICENSE file of the root directory.

import pygame

from physics import compute_velocity, initialize_physics, update_psi
from visualization import initialize_visualization, visualize_scalar, visualize_vector


def main():
    screen = initialize_visualization()

    psi = initialize_physics()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE)):
                pygame.quit()
                exit()

        visualize_scalar(screen, psi)
        u = compute_velocity(psi)
        visualize_vector(screen, u)

        update_psi(psi)

        pygame.display.flip()

if __name__ == "__main__":
    main()