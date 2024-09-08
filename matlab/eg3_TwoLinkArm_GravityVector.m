function gravity_vector = eg3_TwoLinkArm_GravityVector(x)
    global m_link1 m_link2 m_motor2 l1 l2 a1 g;
    theta1 = x(1);
    theta2 = x(2);

    s1 = sin(theta1);
    s12 = sin(theta1+theta2);

    g1 = (m_link1 * l1 + m_motor2 * a1 + m_link2 * a1) * g * s1 ...
        + m_link2 * l2 * g * s12;
    g2 = m_link2 * l2 * g * s12;

    gravity_vector = [g1; g2];
end

