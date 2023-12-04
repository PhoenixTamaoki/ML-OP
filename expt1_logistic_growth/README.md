# Experiment 1: Logistic Growth

This experiment compares the performance of Physics-Informed Neural Networks (PINNs) with and without physics loss (normal neural network) on the logistic growth 1st order Ordinary Differential Equation (ODE).

## Structure of the Neural Network
All experiments are based on a 6-layer neural network with 256 neurons per layer. The Adam optimizer with a learning rate of 0.0005 is utilized.

### 1.1 No Physics Loss
- 700 epochs
- Training data from the whole domain with 20 samples

### 1.2 Physics Loss
- 700 epochs
- Training data from the whole domain with 20 samples

**Remarks:** Both scenarios exhibit similar performance, suggesting sufficient data to capture the complexity of the target function.

### 1.3 No Physics Loss
- 700 epochs
- Training data from the whole domain with 10 samples

### 1.4 Physics Loss
- 700 epochs
- Training data from the whole domain with 10 samples

**Remarks:** The data fits better with the added physics loss regularization, indicating improved fit as the data becomes more sparse.

### 1.5 No Physics Loss
- 700 epochs
- Training data from the whole domain with 5 samples

### 1.6 Physics Loss
- 700 epochs
- Training data from the whole domain with 5 samples (physics loss regularization with weight 0.01, rather than the earlier 0.0001)

**Remarks:** Physics loss significantly improves learning when the weight is increased. Despite concerns of underfitting with larger weights, it appears less significant with less data, and regularization helps fit the data better.

## Comments on Logistic Regression Experiments
Activation function choice is crucial when training PINNs. The tanh activation function initially used led to poor results with physics loss.

### Experiment 1.7
- Tanh activation function
- Model struggles to learn, possibly due to getting stuck at a local optimum or vanishing gradients problem.

### Experiment 1.8
- ReLU activation function
- Reduced domain samples with increased epochs (20000)
- Solution is less accurate but generally follows the correct slope trend.

### Experiment 1.9
- ReLU activation function
- Data loss only
- After 20000 epochs, the neural network follows the data trend without information about the underlying differential equation.

### Experiment 1.10
- ReLU activation function
- Similar behavior to Experiment 1.9 with a smaller sample domain.

### Experiment 1.11
- ReLU activation function
- Physics loss included
- Learned solution is not very accurate, but the algorithm understands to reduce slope.

## Other Comments
- Building and training the model is more challenging and slower than classical methods for the same problem.
- An advantage of PINNs is the ability to recalculate predicted values for any time value after training.
- In practice, PINNs may be more beneficial for solving complex problems that are challenging for traditional methods.
- This project focuses on relatively simple ODEs for pedagogical purposes.