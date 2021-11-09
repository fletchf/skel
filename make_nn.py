from __future__ import print_function

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists

import numpy as np
# import cvxpy as cp
import scipy as sp
# import control
import copy


torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)


# Helper function to construct a linear layer followed by a ReLU nonlinearity
def LinearReLU(in_features, out_features, bias=True):
    dim_hidden = (out_features + in_features) // 2
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        # nn.ReLU6()
        nn.ReLU(),
        # nn.PReLU(),
        nn.BatchNorm1d(out_features)
        # nn.Linear(dim_hidden, out_features, bias=bias)
    )


def LinearTanh(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.Tanh()
    )


def LinearSigmoid(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.Sigmoid()
    )


def LinearBatchNorm(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features)
    )


class NNDenseWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dims, use_skip=True):
        super(NNDenseWithBatchNorm, self).__init__()
        self.dims = [input_dim, *hidden_dims]
        # self.layers_list = [LinearBatchNorm(in_size, out_size)
        #                     for in_size, out_size in zip(self.dims, self.dims[1:])]
        self.layers_list = [nn.Linear(self.dims[0], self.dims[1]),
                            # nn.ReLU(),
                            nn.BatchNorm1d(self.dims[1]),
                            nn.ReLU(),
                            # nn.Tanh(),
                            nn.Linear(self.dims[1], self.dims[2])]
                            # nn.BatchNorm1d(self.dims[2]),
                            # nn.ReLU(),
                            # nn.Tanh()]
        self.layers = nn.Sequential(*self.layers_list)
        self.use_skip = use_skip

    def forward(self, x):
        z = x
        for layer in self.layers_list[:-1]:
            z = layer(z)
        if self.use_skip:
            skip = x @ torch.eye(self.dims[0], m=self.dims[-1])
            out = self.layers_list[-1](z + skip)
        else:
            out = self.layers_list[-1](z)
        return out


class NNDense(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, use_skip=False):
        super(NNDense, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dims = [input_dim, *hidden_dims]
        self.output_dim = output_dim
        self.layers_list = [nn.Sequential(nn.Linear(in_size, out_size), nn.ReLU())
                            for in_size, out_size in zip(self.dims, self.dims[1:])]
        if use_skip:

            out_layer = nn.Linear(self.dims[-1], output_dim - input_dim)
            self.x_bias = nn.Parameter(torch.rand(input_dim))

        else:
            out_layer = nn.Linear(self.dims[-1], output_dim)
            self.C = torch.eye(self.dims[0], m=self.output_dim, device=self.device)

        self.layers_list.append(out_layer)
        self.layers = nn.Sequential(*self.layers_list)
        self.use_skip = use_skip
        

    def forward(self, x):

        if self.use_skip:
            out = torch.hstack([self.layers(x), (x - self.x_bias)])
        else:
            out = self.layers(x) + x @ self.C
        return out


class LinearTest(nn.Module):
    def __init__(self, input_dim):
        super(LinearTest, self).__init__()
        # self.bias = nn.Parameter(torch.rand(1, input_dim))
        # self.x_bias = nn.Parameter(torch.zeros(1, input_dim))
        self.x_bias = torch.zeros(input_dim)
        # self.bias = nn.Parameter(torch.tensor([[0., np.pi, 0., 0.]]))

    def forward(self, x):
        out = x - self.x_bias
        return out


class REN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[20, 20]):
        super(REN, self).__init__()

    def forward(self, x):

        return x