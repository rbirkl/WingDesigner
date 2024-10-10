import pygame
import random
import torch

import numpy as np

from physics import initialize_physics, update_physics, WING_CONTROL_POINTS
from pygame.surface import Surface
from sklearn.model_selection import KFold
from time import time
from torch import from_numpy, load, save, Tensor
from torch.nn import Linear, Module, MSELoss, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict
from visualization import visualize_scalar, visualize_vector

# CONTROL --------------------------------------------------------------------------------------------------------------

# Generate labels and store them on the disk (otherwise they are loaded)?
GENERATE_LABELS = False

# INPUT SPACE ----------------------------------------------------------------------------------------------------------

# The allowed delta range of the parameters of the spline control points in x-direction.
DELTA_RANGE_X = (-50, 50)

# The allowed delta range of the parameters of the spline control points in y-direction.
DELTA_RANGE_Y = (-30, 30)

# ARCHITECTURE ---------------------------------------------------------------------------------------------------------

# The number of hidden layers.
NUMBER_HIDDEN_LAYERS = 2

# The number of neurons per hidden layer.
HIDDEN_SIZE = 16

# LABELS ---------------------------------------------------------------------------------------------------------------

# The number of generated labels.
NUMBER_LABELS = 100

# The maximum number of simulation steps per label.
MAX_STEPS_PER_LABEL = 100

# TRAINING -------------------------------------------------------------------------------------------------------------

# The number of folds for cross-validation.
NUMBER_FOLDS = 2

# The number of epochs.
EPOCHS = 1000

# The batch size.
BATCH_SIZE = 16

# The learning rate.
LEARNING_RATE_TRAINING = 0.001

# OPTIMIZATION ---------------------------------------------------------------------------------------------------------

# The number of particles for particle swarm optimization.
NUMBER_PARTICLES = 16

# The number of optimization steps.
NUMBER_ITERATIONS = 10000

# The weights used for the optimization.
WEIGHTS = (1, 1)

# The learning rate.
LEARNING_RATE_OPTIMIZATION = 0.01

# The attraction of the particle swarm to the currently most optimal particle in [0, 1], where 1 means that the entire
# swarm becomes the best particle.
ATTRACTION_OPTIMIZATION = 0.01

# The number of iterations steps that have to happen before a console log.
ITERATIONS_PER_LOG = 1000

# VISUALIZATION --------------------------------------------------------------------------------------------------------

# The wing color in (red, green, blue).
COLOR_WING = (255, 0, 0)


def generate_labels(screen: Surface) -> [Tensor, Tensor]:
    """
    Generate the labels by running the wing solver for different Bezier parameter values.

    Args:
        screen: The visualization screen.

    Returns:
        The generated inputs and labels.
    """
    if GENERATE_LABELS:
        inputs = []
        labels = []

        for label_index in range(NUMBER_LABELS):
            wing_control_points = []
            for index, wing_control_point in enumerate(WING_CONTROL_POINTS):
                point = wing_control_point
                if (index > 0) and (index < len(WING_CONTROL_POINTS) - 1):
                    x = point[0] + random.uniform(*DELTA_RANGE_X)
                    y = point[1] + random.uniform(*DELTA_RANGE_Y)

                    point = (x, y)
                wing_control_points.append(point)
            wing_control_points = tuple(wing_control_points)

            rho, u, wing_position, wing = initialize_physics(wing_control_points)

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

                wing_position, force = update_physics(rho, u, wing_position, wing)

                pygame.display.flip()

                frame += 1
                end_time = time()
                delta_time = end_time - start_time
                print(f"Label: {label_index}, Frame: {frame}, FPS: {frame / delta_time:.1f}, "
                      f"Force: ({force[0]:.1f}, {force[1]:.1f})")

                if frame == MAX_STEPS_PER_LABEL:
                    input = from_numpy(np.array(wing_control_points[1:-1], dtype=np.float32))
                    label = from_numpy(force)
                    inputs.append(input)
                    labels.append(label)

                    break

        inputs, labels = torch.stack(inputs), torch.stack(labels)

        save(inputs, "inputs.pt")
        save(labels, "labels.pt")
    else:
        inputs = load("inputs.pt", weights_only=True)
        labels = load("labels.pt", weights_only=True)

    return inputs, labels


class MLP(Module):
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize an MLP model.

        Args:
            input_size: The number of input neurons.
            output_size: The number of output neurons.
        """
        super(MLP, self).__init__()

        layers = []
        layers.append(Linear(input_size, HIDDEN_SIZE))
        layers.append(ReLU())

        for _ in range(NUMBER_HIDDEN_LAYERS - 1):
            layers.append(Linear(HIDDEN_SIZE, HIDDEN_SIZE))
            layers.append(ReLU())

        layers.append(Linear(HIDDEN_SIZE, output_size))

        self.__model = Sequential(*layers)

    def forward(self, input: Tensor):
        """
        Perform a forward pass.

        Args:
            input: The model input.
        """
        return self.__model(input)


def average_model_weights(model_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Averages the weights of a list of models.

    Args:
        model_weights: The model weights to be averaged.

    Returns:
        The averaged model weights.
    """
    return {key: torch.mean(torch.stack([model[key] for model in model_weights]), dim=0) for key in
            model_weights[0].keys()}


def train(inputs: Tensor, labels: Tensor) -> Module:
    """
    Train a model.

    Args:
        labels: The labels per input.

    Returns:
        The trained model.
    """
    inputs = inputs.view(inputs.shape[0], -1)

    input_size = inputs.shape[1]
    output_size = labels.shape[1]

    kfold = KFold(n_splits=NUMBER_FOLDS, shuffle=True)
    criterion = MSELoss()

    model_weights = []
    validation_losses = []

    for fold, (training_index, validation_index) in enumerate(kfold.split(inputs)):
        training_inputs, validation_inputs = inputs[training_index], inputs[validation_index]
        training_labels, validation_labels = labels[training_index], labels[validation_index]

        training_dataset = TensorDataset(training_inputs, training_labels)
        training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = MLP(input_size, output_size)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE_TRAINING)

        model.train()
        for epoch in range(EPOCHS):
            for batch_input, batch_label in training_loader:
                output = model(batch_input)
                loss = criterion(output, batch_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Fold: {fold}, Epoch: {epoch}, Loss: {loss.item():.1f}")

        model_weights.append(model.state_dict())

        model.eval()
        with torch.no_grad():
            output = model(validation_inputs)
            loss = criterion(output, validation_labels).item()
            print()
            print(f"Validation loss: {loss:.1f}")
            print()
            validation_losses.append(loss)

    formatted_losses = [f"{loss:.1f}" for loss in validation_losses]
    print(f"Validation losses: {', '.join(formatted_losses)}")
    print()

    averaged_weights = average_model_weights(model_weights)

    averaged_model = MLP(input_size, output_size)
    averaged_model.load_state_dict(averaged_weights)
    averaged_model.eval()

    return averaged_model


def optimize(inputs: Tensor, model: Module) -> Tensor:
    """
    Get the optimized input for a given model with a particle swarm.

    Args:
        inputs: The training inputs.
        model: The wing parameter model.

    Returns:
        The optimized input.
    """
    indices = torch.randperm(inputs.shape[0])[:NUMBER_PARTICLES]
    particles = inputs[indices].view(NUMBER_PARTICLES, -1)

    best_index = None
    for iteration in range(NUMBER_ITERATIONS):
        best_index = None
        min_loss = None

        for index in range(len(particles)):
            particle = particles[index].clone().detach().requires_grad_(True)
            model.zero_grad()

            output = model(particle)
            loss = WEIGHTS[0] * output[0] + WEIGHTS[1] * output[1]

            loss.backward()
            if (best_index is None) or (loss < min_loss):
                best_index = index
                min_loss = loss

            with torch.no_grad():
                particle -= LEARNING_RATE_OPTIMIZATION * particle.grad

            particle.grad.zero_()

            particles[index] = particle.detach()

        for index in range(len(particles)):
            particles[index] += ATTRACTION_OPTIMIZATION * (particles[best_index] - particles[index])

        if iteration % ITERATIONS_PER_LOG == 0:
            print(f"Iteration {iteration}, Loss: {min_loss.item():.4f}")

    print()

    return particles[best_index].view(-1, 2)
