function dxdt = eg1_Linear_Dynamics(t,x,A,mu,sigma_sqrt_upper)
    % x is a column vector
    dxdt = A*x + mu(:) + sigma_sqrt_upper'*randn(size(x));
end

