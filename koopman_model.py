import numpy as np
import scipy.io as spio
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import matplotlib.pyplot as plt

from make_nn import NNDense

torch.set_default_dtype(torch.float64)


class KoopmanModel(nn.Module):

    def __init__(self, input_dim, dim_K, recon=False, hidden_dims=[20, 20]):
        super(KoopmanModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = input_dim
        self.dim_K = dim_K

        use_skip = recon
        self.phi = NNDense(input_dim, dim_K, hidden_dims, use_skip=use_skip)
        if recon:
            self.phi_inv = NNDense(dim_K, input_dim, hidden_dims, use_skip=False)

        self.S = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.M = nn.Parameter(torch.rand(dim_K * 2, dim_K * 2, device=self.device))
        self.epsI = 1e-8 * torch.eye(dim_K * 2, device=self.device)
    
    def forward(self):

        dim = self.dim_K
        S = self.S
        M = self.M

        H = M @ M.T + self.epsI
        F = H[dim:, :dim]
        P = H[dim:, dim:]
        Skew = (S - S.T) / 2
        E = (H[:dim, :dim] + P) / 2 + Skew

        A = torch.linalg.solve(E, F)

        return A


class KoopmanModelOrthogonal(nn.Module):

    def __init__(self, input_dim, dim_K, dt=0.05, recon=False, hidden_dims=[20, 20]):
        super(KoopmanModelOrthogonal, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = input_dim
        self.dim_K = dim_K
        self.dt = torch.tensor(dt, device=self.device)

        self.hidden_layers = NNDense(input_dim, dim_K, hidden_dims, use_skip=True)
        if recon:
            self.reconstructor = NNDense(dim_K, input_dim, hidden_dims, use_skip=False)

        self.S = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
    
    def forward(self):

        S = self.S
        A = torch.matrix_exp((S - S.T) * self.dt)

        return A


class KoopmanModelOrthogonalCT(nn.Module):

    def __init__(self, input_dim, dim_K, recon=False, hidden_dims=[20, 20]):
        super(KoopmanModelOrthogonalCT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = input_dim
        self.dim_K = dim_K

        self.hidden_layers = NNDense(input_dim, dim_K, hidden_dims, use_skip=True)
        if recon:
            self.reconstructor = NNDense(dim_K, input_dim, hidden_dims, use_skip=False)

        self.S = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
    
    def forward(self):

        S = self.S
        A = S - S.T

        return A


class KoopmanModelCT(nn.Module):

    def __init__(self, input_dim, dim_K, recon=False, hidden_dims=[20, 20]):
        super(KoopmanModelCT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = input_dim
        self.dim_K = dim_K

        self.phi = NNDense(input_dim, dim_K, hidden_dims, use_skip=True)
        if recon:
            self.phi_inv = NNDense(dim_K, input_dim, hidden_dims, use_skip=False)

        self.S = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.L = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.Q = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.epsI = 1e-8 * torch.eye(dim_K, device=self.device)
    
    def forward(self):

        S = self.S
        Skew = (S - S.T) / 2
        L = self.L
        Q = self.Q
        M = - L @ L.T - self.epsI

        F = M + Skew
        E = Q @ Q.T + self.epsI

        A = torch.linalg.solve(E, F)

        return A


class KoopmanModelCtrl(nn.Module):

    def __init__(self, dim_x, dim_u, dim_K, dim_v, hidden_dims=[20, 20]):
        super(KoopmanModelCtrl, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_K = dim_K
        self.phi = NNDense(dim_x, dim_K, hidden_dims, use_skip=False)
        self.phi_inv = NNDense(dim_K, dim_x, hidden_dims, use_skip=False)
        self.alpha = NNDense(dim_x + dim_u, dim_v, hidden_dims, use_skip=True)

        self.L = nn.Parameter(torch.rand(dim_K * 2, dim_K * 2, device=self.device))
        self.R = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.B = nn.Parameter(torch.rand(dim_K, dim_v, device=self.device))
        self.epsI = 1e-8 * torch.eye(dim_K * 2, device=self.device)
        self.I = torch.eye(dim_K, device=self.device)


    def forward(self):

        dim = self.dim_K
        L = self.L
        R = self.R
        B = self.B

        M = L @ L.T + self.epsI
        F = M[dim:, :dim]
        P = M[dim:, dim:]
        Skew = (R - R.T) / 2
        E = (M[:dim, :dim] + P) / 2 + Skew

        Acl = torch.linalg.solve(E, F)
        A = (self.I - B @ B.T @ P) @ Acl

        return A