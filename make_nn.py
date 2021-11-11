'''
This script contains a helper function to create dense feedforward neural networks in Pytorch.
'''

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)


class NNDense(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, use_skip=True):
        super(NNDense, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dims = [input_dim, *hidden_dims]
        self.output_dim = output_dim
        self.layers_list = [nn.Sequential(nn.Linear(in_size, out_size), nn.ReLU())
                            for in_size, out_size in zip(self.dims, self.dims[1:])]
        if use_skip:

            out_layer = nn.Linear(self.dims[-1], output_dim)
            self.x_bias = torch.zeros(input_dim)

        else:
            out_layer = nn.Linear(self.dims[-1], output_dim)

        self.layers_list.append(out_layer)
        self.layers = nn.Sequential(*self.layers_list)
        self.use_skip = use_skip
        self.C = torch.eye(self.dims[0], m=self.output_dim, device=self.device)

    def forward(self, x):

        if self.use_skip:
            skip = x @ self.C
            out = skip + self.layers(x)
        else:
            out = self.layers(x)
        return out
