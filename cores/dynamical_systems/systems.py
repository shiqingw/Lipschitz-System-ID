import torch
import torch.nn as nn
import numpy as np

class ZeroDynamics(nn.Module):
    def __init__(self, properties):
        super(ZeroDynamics, self).__init__()
        # Initialize properties
        self.n_state = properties["n_state"]
        self.n_control = properties["n_control"]

    def forward(self, state, action):
        return torch.zeros_like(state)
    
class LinearSystem(nn.Module):
    def __init__(self, properties, params):
        super(LinearSystem, self).__init__()
        # Initialize properties
        self.n_state = properties["n_state"]
        self.n_control = properties["n_control"]

        # Initialize parameters
        self.A = torch.tensor(params["A"])
        self.B = torch.tensor(params["B"])

    def forward(self, state, action):
        return torch.matmul(state, self.A.T) + torch.matmul(action, self.B.T)

class TwoLinkArm(nn.Module):
    def __init__(self, properties, params):
        super(TwoLinkArm, self).__init__()
        # Initialize properties
        self.n_state = properties["n_state"]
        self.n_control = properties["n_control"]

        # Initialize parameters
        self.m_link1 = params["m_link1"]
        self.m_motor1 = params["m_motor1"]
        self.I_link1 = params["I_link1"]
        self.I_motor1 = params["I_motor1"]
        self.m_link2 = params["m_link2"]
        self.m_motor2 = params["m_motor2"]
        self.I_link2 = params["I_link2"]
        self.I_motor2 = params["I_motor2"]
        self.l1 = params["l1"]
        self.l2 = params["l2"]
        self.a1 = params["a1"]
        self.a2 = params["a2"]
        self.kr1 = params["kr1"]
        self.kr2 = params["kr2"]
        self.g = params["g"]
        self.Fv1 = params["Fv1"] # viscous friction joint1 
        self.Fv2 = params["Fv2"] # viscous friction joint2 
        self.Fc1 = params["Fc1"] # Coulomb friction joint1
        self.Fc2 = params["Fc2"] # Coulomb friction joint2
        self.Fc_s1 = params["s1"]
        self.Fc_s2 = params["s2"]

    def forward(self, state, action):
        # Unpack the state and action
        theta1, theta2, dtheta1, dtheta2 = torch.split(state, [1,1,1,1], dim=1)
        tau1, tau2 = torch.split(action, [1,1], dim=1)

        # Constants for the equations
        s1 = torch.sin(theta1)
        c2 = torch.cos(theta2)
        s2 = torch.sin(theta2)
        s12 = torch.sin(theta1 + theta2)

        # Intermediate computations to simplify the equations
        M11 = self.I_link1 + self.m_link1 * self.l1**2 + self.kr1**2 * self.I_motor1 + self.I_link2 \
            + self.m_link2*(self.a1**2 + self.l2**2 + 2 * self.a1 * self.l2 * c2) + self.I_motor2 \
            + self.m_motor2 * self.a1**2
        M12 = self.I_link2 + self.m_link2 * (self.l2**2 + self.a1 * self.l2 * c2) + self.kr2 * self.I_motor2
        M21 = M12  # M21 is symmetric to M12 (M21 == M12)
        M22 = (self.I_link2 + self.m_link2 * self.l2**2 + self.kr2**2 * self.I_motor2)*torch.ones_like(M11)

        h = -self.m_link2 * self.a1 * self.l2 * s2
        C11 = h * dtheta2
        C12 = h * (dtheta1 + dtheta2)
        C21 = -h * dtheta1
        C22 = torch.zeros_like(C11)
        g1 = (self.m_link1 * self.l1 + self.m_motor2 * self.a1 + self.m_link2 * self.a1) * self.g * s1 \
            + self.m_link2 * self.l2 * self.g * s12
        g2 = self.m_link2 * self.l2 * self.g * s12
        
        # The vector b (torque vector minus other terms)
        b1 = tau1 - g1 - C11 * dtheta1 - C12 * dtheta2 - self.Fv1 * dtheta1 - self.Fc1 * torch.tanh(self.Fc_s1*dtheta1)
        b2 = tau2 - g2 - C21 * dtheta1 - C22 * dtheta2 - self.Fv2 * dtheta2 - self.Fc2 * torch.tanh(self.Fc_s2*dtheta2)

        # The inertia matrix
        M = torch.stack((M11, M12, M21, M22), dim=1).reshape(-1, 2, 2)
        
        # The vector of torques minus other terms
        b = torch.cat((b1, b2), dim=1)
        
        # Solve for the accelerations [ddtheta1, ddtheta2]
        ddtheta = torch.linalg.solve(M, b)
        
        return torch.cat((dtheta1, dtheta2, ddtheta), dim=1)
    
    def mass_matrix(self, state):
        # Unpack the state and action
        theta1, theta2, dtheta1, dtheta2 = torch.split(state, [1,1,1,1], dim=1)

        # Constants for the equations
        c2 = torch.cos(theta2)

        # Intermediate computations to simplify the equations
        M11 = self.I_link1 + self.m_link1 * self.l1**2 + self.kr1**2 * self.I_motor1 + self.I_link2 \
            + self.m_link2*(self.a1**2 + self.l2**2 + 2 * self.a1 * self.l2 * c2) + self.I_motor2 \
            + self.m_motor2 * self.a1**2
        M12 = self.I_link2 + self.m_link2 * (self.l2**2 + self.a1 * self.l2 * c2) + self.kr2 * self.I_motor2
        M21 = M12  # M21 is symmetric to M12 (M21 == M12)
        M22 = (self.I_link2 + self.m_link2 * self.l2**2 + self.kr2**2 * self.I_motor2)*torch.ones_like(M11)

        # The inertia matrix
        M = torch.stack((M11, M12, M21, M22), dim=1).reshape(-1, 2, 2)

        return M
    
    def gravity_vector(self, state):
        # Unpack the state and action
        theta1, theta2, dtheta1, dtheta2 = torch.split(state, [1,1,1,1], dim=1)

        # Constants for the equations
        s1 = torch.sin(theta1)
        s12 = torch.sin(theta1 + theta2)

        g1 = (self.m_link1 * self.l1 + self.m_motor2 * self.a1 + self.m_link2 * self.a1) * self.g * s1 \
            + self.m_link2 * self.l2 * self.g * s12
        g2 = self.m_link2 * self.l2 * self.g * s12

        return torch.cat((g1, g2), dim=1)
    
    def coriolis_vector(self, state):
        # Unpack the state and action
        theta1, theta2, dtheta1, dtheta2 = torch.split(state, [1,1,1,1], dim=1)

        # Constants for the equations
        s2 = torch.sin(theta2)

        h = -self.m_link2 * self.a1 * self.l2 * s2
        C11 = h * dtheta2
        C12 = h * (dtheta1 + dtheta2)
        C21 = -h * dtheta1
        C22 = torch.zeros_like(C11)

        return torch.cat((C11 * dtheta1 + C12 * dtheta2, C21 * dtheta1 + C22 * dtheta2), dim=1)


    
class LotkaVolterra(nn.Module):
    def __init__(self, properties, params):
        super(LotkaVolterra, self).__init__()
        # Initialize properties
        self.n_state = properties["n_state"]
        self.n_control = properties["n_control"]

        # Initialize parameters
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.delta = params["delta"]

    def forward(self, state, action):
        # Unpack the state and action
        x, y = torch.split(state, [1,1], dim=1)

        # Compute the derivatives
        dx = self.alpha * x - self.beta * x * y 
        dy = - self.gamma * y + self.delta * x * y

        return torch.cat((dx, dy), dim=1)

class VanDerPol(nn.Module):
    def __init__(self, properties, params):
        super(VanDerPol, self).__init__()
        # Initialize properties
        self.n_state = properties["n_state"]
        self.n_control = properties["n_control"]

        # Initialize parameters
        self.mu = params["mu"]

    def forward(self, state, action):
        # Unpack the state and action
        x, y = torch.split(state, [1,1], dim=1)

        # Compute the derivatives
        dx = y
        dy = self.mu * (1 - x**2) * y - x

        return torch.cat((dx, dy), dim=1)
    
class LorenzSystem(nn.Module):
    def __init__(self, properties, params):
        super(LorenzSystem, self).__init__()
        # Initialize properties
        self.n_state = properties["n_state"]
        self.n_control = properties["n_control"]

        # Initialize parameters
        self.beta = params["beta"]
        self.rho = params["rho"]
        self.sigma = params["sigma"]

    def forward(self, state, action):
        # Unpack the state and action
        x, y, z = torch.split(state, [1,1,1], dim=1)

        # Compute the derivatives
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        return torch.cat((dx, dy, dz), dim=1)