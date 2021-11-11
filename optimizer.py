'''
This script optimizes the simulation error of a Koopman model.
'''

import numpy as np
import torch
from tqdm import trange


class SimErrorOptimizer(object):

    def __init__(self, model, data, data_size, tvec=None, disc=True, errtype='sim', lr=1e-3, lr_decay=0.95, epochs=10000, batch_size=256, opt_tol=1e-6, debug=False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dim_K = model.dim_K
        num_trajs, traj_len = data_size
        X, Xn = data

        self.idx0 = np.repeat(np.cumsum(np.hstack([0, traj_len[:-1]])), traj_len)
        if tvec is None:
            self.tvec = torch.tensor(np.hstack([np.arange(1, T+1) for T in traj_len]), device=self.device)
        else:
            self.tvec = torch.tensor(tvec, device=self.device)
        self.error = errtype        # what error to minimise: 'sim' for simulation error and 'eqn' for one-step prediction error
        self.traj_len = traj_len
        self.num_trajs = num_trajs
        self.disc = disc

        self.X = torch.tensor(X, device=self.device)
        self.Xn = torch.tensor(Xn, device=self.device)
        self.x0 = self.X[self.idx0, :]
        self.data_size = self.idx0.shape[0]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1, line_search_fn='strong_wolfe')
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt_tol = opt_tol      # tolerance for first-order optimality
        self.debug = debug

    def train(self):

        metrics = []

        epochs = trange(self.epochs)

        for epoch in epochs:

            batch_loss, batch_metric = 0., 0.

            for batch in np.array_split(np.random.permutation(self.data_size), max(int(self.data_size / self.batch_size), 1)):

                def sim_loss():

                    Z = self.model.phi(self.X[batch, :])
                    Zn = self.model.phi(self.Xn[batch, :])
                    z0 = self.model.phi(self.x0[batch, :])
                    tvec = self.tvec[batch]

                    mse_loss = torch.nn.MSELoss()
                    A = self.model()

                    # fast matrix power computation as described in Section V-B of https://arxiv.org/abs/2110.06509
                    lam, V = torch.linalg.eig(A)
                    if self.disc:
                        lamt = torch.pow(lam.unsqueeze(-1), tvec).T
                    else:
                        lamt = (lam.repeat(tvec.shape[0], 1) * tvec).exp()
                    At_all = V @ (lamt.diag_embed() @ V.inverse())
                    At = At_all.real.reshape(batch.shape[0], self.dim_K, self.dim_K)
                    Z_sim = (At @ z0.reshape(batch.shape[0], self.dim_K, 1)).reshape(-1, self.dim_K)    # z_t = A^t * z0

                    
                    if self.error == 'sim':
                        loss = mse_loss(Zn, Z_sim)
                    elif self.error == 'eqn':
                        loss = mse_loss(Zn.T, A @ Z.T)


                    if hasattr(self.model, 'phi_inv'):
                        loss += 1e3 * mse_loss(self.model.phi_inv(Z), self.X[batch, :])
                    # metric = 0.5 * torch.linalg.norm(Zn.T - A @ Z.T, ord='fro') ** 2
                    metric = loss

                    return loss, metric
                
                def closure():
                    loss, _ = sim_loss()
                    self.optimizer.zero_grad()
                    loss.backward()
                    return loss

                self.optimizer.step(closure)   

                loss, metric = sim_loss()
                batch_loss += loss
                batch_metric += metric

            batch_loss = batch_loss / max(int(self.data_size / self.batch_size), 1)
            batch_metric = batch_metric / max(int(self.data_size / self.batch_size), 1)

            metrics.append(batch_metric.item())

            epochs.set_description("Loss: %1.4e" % batch_loss.item())

            if (epoch % 1000 == 0):
                for param in self.optimizer.param_groups:
                    param['lr'] *= self.lr_decay    # reduce learning rate

                closure()
                for param in self.model.parameters():
                    if param.grad.abs().max() > self.opt_tol:
                        break   # stop checking grads as optimality condition is not satisfied
                else:   # if for loop finished
                    if self.debug: print('First-order optimality satisfied')
                    break
        else:   # max iterations reached
            if self.debug: print('Max iterations reached')

        return np.hstack(metrics)
