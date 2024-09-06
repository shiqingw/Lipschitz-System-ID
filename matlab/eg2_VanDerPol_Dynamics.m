function dxdt = eg2_VanDerPol_Dynamics(t,state,a,mu,sigma_sqrt_upper)
    x = state(1);
    y = state(2);
    dx = y;
    dy = a*(1-x^2)*y - x;
    dxdt = [dx; dy];
    dxdt = dxdt + mu(:) + sigma_sqrt_upper'*randn(size(state));
end

