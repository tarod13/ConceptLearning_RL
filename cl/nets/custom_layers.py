import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Linear layers stacked together 
class parallel_Linear(nn.Module):
    def __init__(self, n_layers, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.weight = Parameter(torch.Tensor(n_layers, out_features, in_features))
        self.bias = Parameter(torch.Tensor(n_layers, out_features))        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, vars_=None):
        if vars_ is None:
            weight = self.weight
            bias = self.bias
        else:
            weight, bias = vars_
        if len(input.shape) == 2:
            input_shape = 'ik'            
        else:
            input_shape = 'ijk'
        return torch.einsum(input_shape+',jlk->ijl', input, weight) + bias.unsqueeze(0)

    def conditional(self, input, given):
        return torch.einsum('ik,lk->il', input, self.weight[given,:,:]) + self.bias[given,:].unsqueeze(0) 

    def single_output(self, input, label):
        weight = self.weight.data[label,:,:].view(self.out_features, self.in_features)
        bias = self.bias.data[label,:].view(self.out_features)
        output = input.matmul(weight.t()) + bias
        return output      

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



# Linear layer with added noise in weights and biases
class Linear_noisy(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean_weight = Parameter(torch.Tensor(out_features, in_features))
        self.std_weight = Parameter(torch.Tensor(out_features, in_features))
        self.mean_bias = Parameter(torch.Tensor(out_features))
        self.std_bias = Parameter(torch.Tensor(out_features))        
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mean_weight, -bound, bound)
        nn.init.uniform_(self.mean_bias, -bound, bound)
        nn.init.constant_(self.std_weight, 0.5*bound)
        nn.init.constant_(self.std_bias, 0.5*bound)

    def forward(self, input):
        ei = torch.randn(1, self.in_features).to(self.mean_weight.device)
        ej = torch.randn(self.out_features, 1).to(self.mean_weight.device)
        ewij = torch.sign(ei)*torch.sign(ej)*((ei).abs().pow(0.5))*((ej).abs().pow(0.5))
        ebj = (torch.sign(ej)*((ej).abs().pow(0.5))).squeeze(1)
        weight = self.mean_weight + self.std_weight * ewij
        bias = self.mean_bias + self.std_bias * ebj
        return torch.einsum('ik,lk->il', input, weight) + bias.unsqueeze(0) 
        
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        ) 