A_sys = [-0.1, 2.0;
        -2.0, -0.1];
mu = [0, 0];
sigma = 10*eye(2);
sigma_sqrt_upper = chol(sigma, 'upper');

x0 = [3,3];
tspan = 0:0.1:10;
[t,x] = ode45(@(t,x) eg1_Linear_Dynamics(t,x,A_sys,mu,sigma_sqrt_upper),tspan,x0);
plot(x(:,1), x(:,2));

