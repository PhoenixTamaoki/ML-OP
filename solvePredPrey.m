alpha = 2; beta = 4/3; delta = 10; gamma = 1; tf = 10; 
tspan = [0 tf];
f = @(t,y) [alpha*y(1)-beta*y(1)*y(2); delta*y(1)*y(2)-gamma*y(2)];
[t,u] = ode45(f,[0 1],[10,10]);
plot(t,u(:,1),'b-',t,u(:,2),'r-');

%Take random points and add some noise
len = ceil(size(u,1)/3);
indices = randi([1,len],ceil(len/2),1);
noise = 0.1*randn(ceil(len/2),2);
data = u(indices,:) + noise;

%Write to text file
csvwrite('pred_prey_true.txt',u)
csvwrite('pred_prey_noisy_data.txt',data)