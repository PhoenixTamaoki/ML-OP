# ML Optimization Group Project Repository

## Team Members
- **Phoenix Tamaoki (tamaop)**
  - Email: `&#116;&#97;&#109;&#97;&#111;&#112;&#64;&#114;&#112;&#105;&#46;&#101;&#100;&#117;`

- **Jichuan Wu (wuj20)**
  - Email: `&#119;&#117;&#106;&#50;&#48;&#64;&#114;&#112;&#105;&#46;&#101;&#100;&#117;`

- **Yuyang Gong (gongy2)**
  - Email: `&#103;&#111;&#110;&#103;&#121;&#50;&#64;&#114;&#112;&#105;&#46;&#101;&#100;&#117;`

- **Calvin Ang (angc2)**
  - Email: `&#97;&#110;&#103;&#99;&#50;&#64;&#114;&#112;&#105;&#46;&#101;&#100;&#117;`

## Project Overview

Welcome to the ML Optimization Group Project repository! Our team, consisting of Phoenix Tamaoki, Jichuan Wu, Yuyang Gong, and Calvin Ang, is focused on exploring and discussing the Physically Informed Neural Networks (PINNs) model and comparing it to conventional Neural Networks (NN).

## Repository Contents
This repository contains the problem set and three experiments we have conducted, which are in the presentation slides, and lecture notes. See lecture notes for more details about the experiments.

## Problem Set
The problem set is available in the root directory of this repository. It contains the following files:
### Problem Set
- `problem_set.pdf`: PDF file containing the problem set.
- `problem_set.tex`: LaTeX file generates the PDF file.

### Problem Set solutions
- `pset_sol.ipynb`: Jupyter notebook containing the solutions to the problem set.
- `pset_nn_sol.jpeg`: Picture of the output solution for the NN without physics loss
- `PINNs_output.jpeg`: Picture of the output solution for the PINNs algorithm

### Problem Set experiments
- `pset_experiment.py`: Python script containing the code for the experiments we do for the problem set. It just have the code for the experiments, not the code for the solutions. The code for the solutions is in the Jupyter notebook `pset_sol.ipynb`.

## Experiment Structure
All experiments follow a consistent structure, utilizing a 5-layer neural network with 256 neurons per layer. The Adam optimizer with a learning rate of 0.0005 is employed for training.

## Experiment Highlights

### Experiment 1: Logistic Growth
Investigates the performance of PINNs with and without physics loss on a 1st order ODE, showcasing the impact of data sparsity and the role of activation functions.

### Experiment 2: Projectile Motion
Compares PINNs with and without physics loss on a 2nd order ODE governing projectile motion. Demonstrations highlight the accuracy of PINNs in predicting physically feasible flight paths.

### Experiment 3: Predator-Prey Model
Examines PINNs' effectiveness on the Lotka-Volterra equations, modeling predator-prey population dynamics. Future experiments will focus on parameter tuning and solving inverse problems.

## Key Findings
- PINNs demonstrate versatility in capturing the complexity of different ODEs.
- Activation function choice is pivotal in training PINNs effectively.
- PINNs exhibit accuracy in solving projectile motion problems, even with relatively few epochs.
- Further exploration is planned for the predator-prey model, with a focus on parameter tuning and solving inverse problems.
