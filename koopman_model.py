'''
This script defines the Koopman model described in https://arxiv.org/abs/2110.06509
'''

import torch
from torch._C import device
import torch.nn as nn

from make_nn import NNDense

torch.set_default_dtype(torch.float64)


class KoopmanModel(nn.Module):

    def __init__(self, input_dim, dim_K, recon=True, hidden_dims=[20, 20]):
        super(KoopmanModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = input_dim
        self.dim_K = dim_K

        self.phi = NNDense(input_dim, dim_K, hidden_dims, use_skip=True)  # neural net for Koopman mapping phi
        if recon:
            self.phi_inv = NNDense(dim_K, input_dim, hidden_dims, use_skip=False) # neural net for inverse of phi

        self.R = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))     # parameter for Koopman matrix A
        self.L = nn.Parameter(torch.rand(dim_K * 2, dim_K * 2, device=self.device)) # parameter for Koopman matrix A
        self.epsI = 1e-8 * torch.eye(dim_K * 2, device=self.device)
    
    # forward function only computes Koopman matrix A
    def forward(self):

        dim = self.dim_K
        R = self.R
        L = self.L

        M = L @ L.T + self.epsI
        F = M[dim:, :dim]
        P = M[dim:, dim:]
        Skew = (R - R.T) / 2
        E = (M[:dim, :dim] + P) / 2 + Skew

        A = torch.linalg.solve(E, F)

        return A


class KoopmanModelCT(nn.Module):

    def __init__(self, input_dim, dim_K, recon=True, hidden_dims=[20, 20]):
        super(KoopmanModelCT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = input_dim
        self.dim_K = dim_K

        self.hidden_layers = NNDense(input_dim, dim_K, hidden_dims, use_skip=True)
        if recon:
            self.reconstructor = NNDense(dim_K, input_dim, hidden_dims, use_skip=False)

        self.R = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.Q = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.N = nn.Parameter(torch.rand(dim_K, dim_K, device=self.device))
        self.epsI = 1e-8 * torch.eye(dim_K, device=self.device)
    
    def forward(self):

        S = self.S
        Q = self.Q
        N = self.N

        Skew = (S - S.T) / 2
        F = Skew - Q @ Q.T - self.epsI
        E = N @ N.T + self.epsI

        A = torch.linalg.solve(E, F)

        return A