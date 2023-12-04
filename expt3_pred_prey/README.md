# Experiment 3: Predator-Prey Model

This experiment assesses the performance of Physics-Informed Neural Networks (PINNs) with and without physics loss (normal neural network) on the predator-prey model using the Lotka-Volterra equations.

## Structure of the Neural Network
Similar to Experiment 1, the neural network architecture comprises 6 layers with 256 neurons per layer. The Adam optimizer is employed with a learning rate of 0.0005.

### Lotka-Volterra Equation for Population Dynamics

The Lotka-Volterra Equation models the population dynamics of an ecosystem with a prey species and a predator species. The code draft for this experiment is available in our pred_prey_PINN.ipynb file. However, this code requires parameter tuning for improved results. 

We intend to conduct similar experiments as described above with this Ordinary Differential Equation (ODE). Additionally, we plan to execute an experiment showcasing PINNs' capability to solve inverse problems.