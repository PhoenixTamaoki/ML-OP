import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#Make true solution

#Make the PINNs solution
#Use https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563 and convert to pytorch
# import torch
# import torch.nn as nn
# import torch.optim as optim

#Define custom loss function

class Net(torch.nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

from torch.func import functional_call, grad, vmap

model = Net(1, 2, 3, 3)

# # calls the neural network forward function using time to give a prediction 
# def f(x: torch.Tensor, params: dict[str, torch.nn.Parameter]) -> torch.Tensor:
#     return functional_call(model, params, (x, ))

# # calculates gradients 
# dfdt = vmap(grad(f), in_dims=(0, None))


# loss function takes in the predicted output and the actual output
#x is prey population, y is predator population, t time
def custom_loss(prey_samples, pred_samples, p_prey, p_pred, t_phys):
    #Data loss term
    alpha = 2 
    beta = 4/3
    delta = 4
    gamma = 3 

    loss_data = nn.MSELoss(prey_samples, p_prey) + nn.MSELoss(pred_samples, p_pred)
    
    prey_phys, pred_phys = model(t_phys)
    dprey = torch.autograd.grad(prey_phys, t_phys, torch.ones_like(prey_phys), create_graph=True)[0]
    dpred = torch.autograd.grad(pred_phys, t_phys, torch.ones_like(pred_phys), create_graph=True)[0]
    phys_loss_prey = nn.MSELoss(dprey, alpha * prey_phys+ beta * prey_phys * pred_phys)
    phys_loss_pred = nn.MSELoss(dpred, gamma * prey_phys * pred_phys + delta * pred_phys)

    return loss_data + phys_loss_prey + phys_loss_pred

# Create an instance of your neural network
net = Net(1, 2, 3, 3)

# Define the data, t_train, s_train, and t_phys
# Convert them to PyTorch tensors if they are not already

# Define data and physics loss weights
#data_weight = 1.0
#phys_weight = 1.0

# Create an optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Define mu as a learnable parameter
#mu = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

# placeholders for when we implement the nn with out actual data -------------------------------------------------------------------------------------
# number of training iterations
num_iterations =  100 

# the input features for the neural net
t_train = np.zeros(100) 

# training data for the target variable, which is displacement. Specifically, it's a numerical 
# data structure (e.g., an array, tensor, or similar) containing the observed or measured 
# displacement values corresponding to the input time values t_train
t_phys = torch.tensor(np.linspace(0,30,100))
file_path = 'pred_prey_noisy_data.txt'
prey_train_list = []
pred_train_list = []
time_list = []
try:
    with open(file_path, 'r') as file:
        for line in file:
            # Process each line (line is a string) time, prey, pred
            values = line.strip().split(',')
            time_list.append(values[0])
            prey_train_list.append(values[1])
            pred_train_list.append(values[2])
except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
prey_train = torch.tensor(prey_train_list)
pred_train = torch.tensor(pred_train_list)
time = torch.tensor(time_list)


# Training loop
for i in range(num_iterations):
    p_prey,p_pred = model(t_train)
    #prey_train and pred_train are data from github
    loss = custom_loss(prey_train, pred_train, p_prey, p_pred, t_phys)
    loss.backward()
    optimizer.step()
# the array of time values that we want to predict
# t_phys = torch.linspace(0, 1, steps=30, requires_grad=True)



# # Access the learned mu value
# learned_mu = mu.item()

import matplotlib.pyplot as plt

t_eval = torch.tensor(np.linspace(0,300))
prey_eval,pred_eval = model(t_eval)
plt.plot(t_eval, prey_eval,'b-' , t_eval, pred_eval, 'r-', time, prey_train, 'bx', time, pred_train, 'ro')
plt.show()