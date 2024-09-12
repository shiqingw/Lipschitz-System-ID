function M = eg3_TwoLinkArm_Mass(x)
    global m_link1 m_motor1 I_link1 I_motor1 m_link2 m_motor2 I_link2 ...
    I_motor2 l1 l2 a1 a2 kr1 kr2 g Fv1 Fv2;
    theta1 = x(1); % origin shifted to the downward position
    theta2 = x(2);

    c2 = cos(theta2);

    M11 = I_link1 + m_link1 * l1^2 + kr1^2 * I_motor1 + I_link2 ...
            + m_link2*(a1^2 + l2^2 + 2 * a1 * l2 * c2) + I_motor2 ...
            + m_motor2 * a1^2;
    M12 = I_link2 + m_link2 * (l2^2 + a1 * l2 * c2) + kr2 * I_motor2;
    M21 = M12;
    M22 = (I_link2 + m_link2 * l2^2 + kr2^2 * I_motor2);

    M = [M11, M12; M21, M22];
    
end