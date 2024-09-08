function dxdt = eg3_TwoLinkArm_Dynamics_with_Input(t,x,tau)
    global m_link1 m_motor1 I_link1 I_motor1 m_link2 m_motor2 I_link2 ...
    I_motor2 l1 l2 a1 a2 kr1 kr2 g Fv1 Fv2;
    theta1 = x(1); % origin shifted to the downward position
    theta2 = x(2);
    dtheta1 = x(3);
    dtheta2 = x(4);

    c2 = cos(theta2);
    s2 = sin(theta2);

    M11 = I_link1 + m_link1 * l1^2 + kr1^2 * I_motor1 + I_link2 ...
            + m_link2*(a1^2 + l2^2 + 2 * a1 * l2 * c2) + I_motor2 ...
            + m_motor2 * a1^2;
    M12 = I_link2 + m_link2 * (l2^2 + a1 * l2 * c2) + kr2 * I_motor2;
    M21 = M12;
    M22 = (I_link2 + m_link2 * l2^2 + kr2^2 * I_motor2);

    h = -m_link2 * a1 * l2 * s2;
    C11 = h * dtheta2;
    C12 = h * (dtheta1 + dtheta2);
    C21 = -h * dtheta1;
    C22 = 0;
    
    gravity_vector = eg3_TwoLinkArm_GravityVector(x);
    g1 = gravity_vector(1);
    g2 = gravity_vector(2);

    tau1 = tau(1);
    tau2 = tau(2);
    
    b1 = tau1 - g1 - C11 * dtheta1 - C12 * dtheta2 - Fv1 * dtheta1;
    b2 = tau2 - g2 - C21 * dtheta1 - C22 * dtheta2 - Fv2 * dtheta2;

    M = [M11, M12; M21, M22];
    b = [b1; b2];
%     disp(M);
%     disp(b);
%     disp([C11,C12;C21,C22]);
%     disp([g1;g2]);

    ddtheta = M\b;

    dxdt = [dtheta1; dtheta2; ddtheta];
end