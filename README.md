# Wing Designer

An AI-based designer of optimal airplane wings.

## Demo

https://github.com/user-attachments/assets/eac481ff-720d-4853-86ed-15046c83884a

![Optimal_Wing](https://github.com/user-attachments/assets/743016b0-5a8d-459c-8176-b5294286fefb)

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation

To set up the environment for this project, you'll need to create a conda environment using the
provided `environment.yml` file. Follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rbirkl/WingDesigner.git
   cd WingDesigner

2. **Create the conda environment**:
   ```bash
    conda env create -f environment.yml

3. **Activate the conda environment**:
   ```bash
    conda activate WingDesigner

## Usage

To run the program, execute the main.py script:

```bash
python main.py
```

The following features are displayed:

- Scalar field = fluid density
- Vector field = fluid velocity
- Red polygon = airplane wing

When the program is started, it first trains a surrogate model and then optimizes the wing via an AI-based particle
swarm.

## Features

The following features are supported by Wing Designer

- Real-time visualizer of physical fields via Pygame
- Fast field computations via CUDA
- 4th-order Runke-Kutta solver of compressible Navier-Stokes equations with gravitation
- Parametrization of wing via Bezier splines
- Generation of label data via multiple simulation runs
- Trainer of surrogate MLP-model with cross-validation
- Wing optimization via particle swarm based gradient descent optimization

Note that theory.txt contains a short theoretical derivation of the numerical equations used.

## License

The license is MIT, see the ```LICENSE``` file.
