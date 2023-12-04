alpha = 2; beta = 4/3; delta = 4; gamma = 3; tf = 20; 
%alpha = 4; beta = 0.4; delta = 0.9; gamma = 0.3; tf = 20; 
tspan = [0 tf];
f = @(t,y) [alpha*y(1)-beta*y(1)*y(2); delta*y(1)*y(2)-gamma*y(2)];
[t,u] = ode45(f,[0 tf],[2 1]);
%[t,u] = ode45(f,[0 tf],[10 10]);
plot(t,u(:,1),'b-',t,u(:,2),'r-');
hold on;

%Take random points and add some noise
len = ceil(size(u,1)/10);
indices = randi([1,len],ceil(len/2),1);
noise = 0.1*(randn(ceil(len/2),2)-0.5);
data = u(indices,:) + noise;
plot(t(indices),data(:,1),'bx', t(indices), data(:,2), 'ro');
title('Predator Prey Model');
xlabel('Time');
ylabel('Population');
legend('True Prey Population', 'True Predator Population', 'Noisy Prey Data', 'Noisy Predator Data')

%Write to text file
csvwrite('pred_prey_true.txt',[t,u])
csvwrite('pred_prey_noisy_data.txt',[t(indices),data])