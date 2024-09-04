import torch 
import torch.nn as nn 
from .layers import SandwichFc, SandwichLin

def get_activation(activation_name):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'identity':
        return nn.Identity()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class NeuralNetwork(nn.Module):
    def __init__(self, config, input_bias=None, input_transform=None, output_transform=None, train_transform=False, zero_at_zero=False):
        super().__init__()
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.gamma = config.gamma 
        self.layer = config.layer
        self.activations = config.activations  # List of activation function names
        self.widths = config.widths  # List of widths for each layer
        self.zero_at_zero = zero_at_zero

        if input_bias is None:
            self.input_bias = torch.nn.Parameter(torch.zeros(self.in_features, dtype=torch.float32), requires_grad=False)
        else:
            self.input_bias = torch.nn.Parameter(torch.tensor(input_bias), requires_grad=bool(train_transform))

        if input_transform is None:
            self.input_transform = torch.nn.Parameter(torch.ones(self.in_features, dtype=torch.float32), requires_grad=False)
        else:
            self.input_transform = torch.nn.Parameter(torch.tensor(input_transform), requires_grad=bool(train_transform))

        if output_transform is None:
            self.output_transform = torch.nn.Parameter(torch.ones(self.out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.output_transform = torch.nn.Parameter(torch.tensor(output_transform), requires_grad=bool(train_transform))

        if len(self.activations) != len(self.widths) - 2:
            raise ValueError("Number of activations must be two less than number of widths. The last layer has no activation.")
        if self.widths[-1] != self.out_features:
            raise ValueError("Last width must match number of output channels.")
        if self.widths[0] != self.in_features:
            raise ValueError("First width must match number of input channels.")

        if self.layer == 'Plain':
            layers = []
            for i in range(len(self.activations)):
                layers.append(nn.Linear(self.widths[i], self.widths[i + 1], bias=True))
                layers.append(get_activation(self.activations[i]))
            layers.append(nn.Linear(self.widths[-2], self.out_features))  # Final layer without activation
            self.model = nn.Sequential(*layers)

        elif self.layer == 'Sandwich':
            layers = []
            for i in range(len(self.activations)):
                scale = 1.0
                layers.append(SandwichFc(self.widths[i], self.widths[i + 1], bias=True, activation=self.activations[i], scale=scale))
            layers.append(SandwichLin(self.widths[-2], self.out_features, bias=True, scale=self.gamma, AB=False))  # Last layer with identity activation
            self.model = nn.Sequential(*layers)
        
        elif self.layer == 'Lip_Reg':
            layers = []
            for i in range(len(self.activations)):
                layers.append(nn.Linear(self.widths[i], self.widths[i + 1], bias=True))
                layers.append(get_activation(self.activations[i]))
            layers.append(nn.Linear(self.widths[-2], self.out_features))  # Final layer without activation
            self.model = nn.Sequential(*layers)
       
        else:
            raise ValueError(f"Unsupported layer: {self.layer}")
            
    def forward(self, x_in):
        input_transform = torch.diag(self.input_transform)
        x = torch.mm(x_in-self.input_bias, input_transform)
        model_output = self.model(x)
        output_transform = torch.diag(self.output_transform)
        out = torch.mm(model_output, output_transform)

        if self.zero_at_zero:
            zeros = torch.zeros_like(x_in)
            zeros = torch.mm(zeros-self.input_bias, input_transform)
            zero_values = self.model(zeros)
            zero_values = torch.mm(zero_values, output_transform)
            out = out - zero_values
            
        return out