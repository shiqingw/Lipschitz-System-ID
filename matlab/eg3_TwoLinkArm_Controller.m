function u = eg3_TwoLinkArm_Controller(t, x)
    global qd Kp Kd am1 am2 freq1 freq2 th1 th2;
    theta1 = x(1);
    theta2 = x(2);
    dtheta1 = x(3);
    dtheta2 = x(4);

    gravity =  eg3_TwoLinkArm_GravityVector(x);
    coriolis = eg3_TwoLinkArm_Coriolis(x);
    mass = eg3_TwoLinkArm_Mass(x);
    
    % q0 = qd + 0.2 * sin(2*pi*t) * ones(size(qd, 1),1);
    % dq0 = 0.2 * 2*pi* cos(2*pi*t) * ones(size(qd, 1),1);
    % q_tilde = [theta1; theta2] - q0;
    % dtheta = [dtheta1; dtheta2] - dq0;
    q_tilde = [theta1; theta2] - qd;
    dtheta = [dtheta1; dtheta2];
    u = gravity + coriolis + mass* (- Kp*q_tilde - Kd*dtheta);
    u(1) = u(1) + am1 * sin(2*pi*freq1*t + th1);
    u(2) = u(2) + am2 * sin(2*pi*freq2*t + th2);
end