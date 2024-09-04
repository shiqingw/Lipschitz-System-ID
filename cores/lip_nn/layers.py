import torch
import torch.nn as nn
import torch.nn.functional as F

## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

def get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'identity':
        return lambda x: x
    else:
        raise ValueError("Unsupported activation function: {}".format(activation))

## from https://github.com/acfr/LBDN
class SandwichFc(nn.Linear): 
    def __init__(self, in_features, out_features, bias=True, activation='relu', scale=1.0):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm() 
        self.scale = scale 
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))   
        self.Q = None
        self.activation_str = activation
        self.activation = get_activation_fn(activation)
    
    def forward(self, x): # Eq. (9)
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach() # Q = [A^T, B^T]
        x = F.linear(self.scale * x, Q[:, fout:]) # B*h 
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = self.activation(x) * torch.exp(self.psi) # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
        return x
    
class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale   
        self.AB = AB
        self.Q = None

    def forward(self, x): # Eq. (9)
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B @ x 
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T) # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x
