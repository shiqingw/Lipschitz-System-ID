function coriolis = eg3_TwoLinkArm_Coriolis(x)
    global m_link1 m_motor1 I_link1 I_motor1 m_link2 m_motor2 I_link2 ...
    I_motor2 l1 l2 a1 a2 kr1 kr2 g Fv1 Fv2;
    theta1 = x(1); % origin shifted to the downward position
    theta2 = x(2);
    dtheta1 = x(3);
    dtheta2 = x(4);

    c2 = cos(theta2);
    s2 = sin(theta2);

    h = -m_link2 * a1 * l2 * s2;
    C11 = h * dtheta2;
    C12 = h * (dtheta1 + dtheta2);
    C21 = -h * dtheta1;
    C22 = 0;

    coriolis = [C11 * dtheta1 + C12 * dtheta2;
                C21 * dtheta1 + C22 * dtheta2];
end