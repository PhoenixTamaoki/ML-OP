import numpy as np
import torch

#Make true solution

#Make the PINNs solution
#Use https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563 and convert to pytorch
# import torch
# import torch.nn as nn
# import torch.optim as optim

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define your neural network architecture here

    def forward(self, t):
        a = 10
        # Implement the forward pass of your neural network
        # to predict displacement

# Create an instance of your neural network
net = Net()

# Define the data, t_train, s_train, and t_phys
# Convert them to PyTorch tensors if they are not already

# Define data and physics loss weights
data_weight = 1.0
phys_weight = 1.0

# Create an optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Define mu as a learnable parameter
mu = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

# placeholders for when we implement the nn with out actual data -------------------------------------------------------------------------------------
# number of training iterations
num_epochs =  100 

# the input features for the neural net
t_train = np.zeros(100) 

# training data for the target variable, which is displacement. Specifically, it's a numerical 
# data structure (e.g., an array, tensor, or similar) containing the observed or measured 
# displacement values corresponding to the input time values t_train
s_train = np.zeros(100) 

# the array of time values 
t_phys = np.zeros(100)

# Training loop
for epoch in range(num_epochs):
    # Data loss
    s_train_hat = net(t_train)
    data_loss = torch.mean((s_train - s_train_hat)**2)
    
    
    # Physics loss
    s_phys_hat = net(t_phys)
    s_x = s_phys_hat[:, 0]
    s_y = s_phys_hat[:, 1]
    v_x = torch.autograd.grad(s_x, t_phys, create_graph=True)[0]
    v_y = torch.autograd.grad(s_y, t_phys, create_graph=True)[0]
    a_x = torch.autograd.grad(v_x, t_phys, create_graph=True)[0]
    a_y = torch.autograd.grad(v_y, t_phys, create_graph=True)[0]
    v = torch.cat([v_x, v_y], dim=1)
    a = torch.cat([a_x, a_y], dim=1)
    v = v.detach()  # Stop gradients for velocity
    speed = torch.norm(v, dim=1, keepdim=True)
    g = torch.tensor([[0.0, 9.81]])
    phys_loss = torch.mean(((-mu * speed * v - g - a)**2))
    
    # Total loss
    loss = data_weight * data_loss + phys_weight * phys_loss
    
    # Gradient step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Access the learned mu value
learned_mu = mu.item()


#Solve with RK4 solver

#compare accracy