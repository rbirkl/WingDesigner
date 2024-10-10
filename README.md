# Wing Designer

An AI-based designer of optimal airplane wings.

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

- Real-time Visualizer of physical fields via Pygame
- Fast processing of field computations via CUDA
- 4th-order Runke-Kutta solver of compressible Navier-Stokes equations
- Parametrization of wing via Bezier splines
- Generation of label data via multiple simulation runs
- Trainer of surrogate MLP-model
- Wing optimization via particle swarm based gradient descent optimization

Note that theory.txt contains a short theoretical derivation of the numerical equations used.

## License

The license is MIT, see the ```LICENSE``` file.