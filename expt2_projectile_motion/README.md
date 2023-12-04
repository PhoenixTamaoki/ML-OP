# Experiment 2: Projectile Motion

This experiment evaluates the performance of Physics-Informed Neural Networks (PINNs) with and without physics loss (normal neural network) on the projectile motion 2nd order Ordinary Differential Equation (ODE).

## Structure of the Neural Network
Similar to Experiment 1, the neural network consists of 6 layers with 256 neurons per layer. The Adam optimizer is used with a learning rate of 0.0005.

### Experiments for Projectile Motion with Demonstration

These experiments are presented in our problem set, as they showcase the accuracy of our PINNs algorithm in solving projectile motion problems.

- The code for these experiments is available in pset_nn_sol.jpeg.
- Results for the neural network without physics loss are illustrated in normal_nn_output.jpeg.
- Results for the PINNs algorithm are provided in PINNs_output.jpeg.

Observations:
- The neural network with only data loss did not produce a physically feasible flight path.
- In contrast, the PINNs algorithm achieved a fairly accurate flight path with relatively few epochs.