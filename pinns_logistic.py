import numpy as np
import torch

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE
def log_growth(y, t):
    r = 0.1
    K = 100
    return r*y*(1 - (y/K))

# Initial condition
y0 = 10
r = 0.1
K = 100

# Time points
t = np.linspace(0, 100,500)

# Solve the ODE
solution = odeint(log_growth, y0, t)

#start = solution[0]
#end = solution[499]

# Generate training data (based on noisy data or the true solution)
# training_t, training_y(prey, predator) noisy with gaussian, true_solution
pool = range(0,500)
idx = np.random.choice(pool, 10, replace=True)
training_t = t[idx]
# reshape to (100, 1)
training_t = training_t[:, np.newaxis]
true_data = solution[idx]
noise = np.random.normal(-0.1, 0.1, true_data.shape)
training_y = true_data + noise

# plot the training datas and the true solution
plt.figure(figsize=(8, 6))
plt.plot(t, solution, 'r-')
plt.scatter(training_t, training_y, label='Prey (x)')
plt.title('Logistic Growth')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the PINN model
class PINNModel(nn.Module):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.dense1 = nn.Linear(1, 256)  # Input: time
        #self.dense2 = nn.Linear(50, 50)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 256)
        self.dense4 = nn.Linear(256, 256)
        self.dense5 = nn.Linear(256, 256)
        #self.dense6 = nn.Linear(256, 1)  # Output: prey, predator
        #For trying to learn the growth rate R
        self.dense6 = nn.Linear(256,2)
    def forward(self, t):
        t = t.view(-1,1)
        # x = torch.tanh(self.dense1(t))
        # x = torch.tanh(self.dense2(x))
        # x = torch.tanh(self.dense3(x))
        # x = torch.tanh(self.dense4(x))
        # x = torch.tanh(self.dense5(x))
        # sol = torch.tanh(self.dense6(x))
        relu = torch.nn.ReLU()
        x = relu(self.dense1(t))
        x = relu(self.dense2(x))
        x = relu(self.dense3(x))
        x = relu(self.dense4(x))
        x = relu(self.dense5(x))
        sol = self.dense6(x)
        return sol
    
#Convert training data to np ndarray
if (type(training_t) is np.ndarray):
    training_t = torch.tensor(training_t, requires_grad=True, dtype=torch.float32)
    training_y = torch.tensor(training_y, requires_grad=True, dtype=torch.float32)
    true_data = torch.tensor(true_data, dtype=torch.float32)


# # Instantiate the PINN model and define optimizer
model = PINNModel()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
#optimizer = optim.Adam(model.parameters(), lr=0.0005)
#optimizer = optim.LBFGS(model.parameters())

#grid to test phys loss
t_phys = torch.linspace(0,100,100, requires_grad=True)

# # Training loop
num_epochs = 700  # Adjust as needed
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions_data = model(training_t)
    predictions_data = predictions_data.requires_grad_()
    data_loss = torch.mean((training_y - predictions_data)**2)
    # print(prey_data.shape)
    # print(pred_data.shape)
    # print(prey_sample.shape)
    # print(pred_sample.shape)

    predictions_phys = model(t_phys)
    predictions_phys = predictions_phys.requires_grad_()
    dPdt = torch.autograd.grad(predictions_phys, t_phys, grad_outputs=torch.ones_like(predictions_phys), create_graph=True)[0]
    
    phys_loss = torch.mean((dPdt - r*predictions_phys*(1 - (predictions_phys/K)))**2)
    #n = predictions_phys.size(dim=0)-1
    #boundary_loss = (start - predictions_phys[0].item())**2 + (end - predictions_phys[n].item())**2
    loss = data_loss + 0.01*phys_loss #+ 10*boundary_loss.item()

    # if epoch % 500 == 0:
    #     print(epoch)
    #     print(f'Epoch {epoch}, Loss: {loss.item()}')
    #     #print(f'Physics Loss: {phys_loss}')
    #     print(f'Data Loss: {data_loss}')
    #     plt.plot(training_t.detach().numpy(), training_y.detach().numpy(),'bx')
    #     plt.plot(t_phys.detach().numpy(), predictions_phys.detach().numpy(), 'g--')
    #     # plt.plot(training_t.detach().numpy(), training_y.detach().numpy(), 'mD')
    #     plt.plot(training_t.detach().numpy(), predictions_data.detach().numpy(), 'g+')
    #     plt.plot(t, solution, 'r-')
    #     plt.show()

    loss.backward()

    optimizer.step()

test_t = np.linspace(0, 100, 1000)
test_t = test_t[:, np.newaxis]
test_t = torch.tensor(test_t, requires_grad=True, dtype=torch.float32)

results = model.forward(test_t)
# test_y = torch.split(results, 1, dim=1)

# plt.plot(t, solution[:, 0], label='Prey (x)')
# plt.plot(t, solution[:, 1], label='Predator (y)')
# plt.plot(test_t.detach().numpy(), test_y.detach().numpy()[:, 0], label='Prey (x)')
# plt.plot(test_t.detach().numpy(), test_y.detach().numpy()[:, 1], label='Predator (y)')
# plt.title('Lotka-Volterra Model')
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.plot(training_t.detach().numpy(),training_y.detach().numpy(), 'bx')
plt.plot(t, solution, 'r-')
plt.plot(test_t.detach().numpy(), results.detach().numpy(), 'g--')
plt.title('Logistic Growth with Physics Loss')
plt.xlabel('Time')
plt.savefig('experiment1_7.jpeg', format = 'jpeg')
plt.show()