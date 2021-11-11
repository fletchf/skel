'''
This script loads and evaluates models trained by lasa_train.py
Used to produce Figure 3 in https://arxiv.org/abs/2110.06509
'''

import numpy as np
import scipy.io as spio
import torch
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# function for simulating discrete-time Koopman model for given initial condition
def forward_sim_disc(model, num_timesteps, init_cond, scale):

    with torch.no_grad():
        tvec = torch.arange(0, int(num_timesteps)).unsqueeze(-1)
        z0 = model.phi(torch.tensor((init_cond / scale).T, device=device))
        A = model()
        lam, V = torch.linalg.eig(A)
        Lamt = torch.pow(lam.repeat(tvec.shape[0], 1), tvec)
        At = V @ (Lamt.diag_embed() @ V.inverse())
        Z_sim = (At.real @ z0.T).squeeze()
        if hasattr(model, 'phi_inv'):
            X_sim = model.phi_inv(Z_sim) * scale.T
        else:
            X_sim = (Z_sim + model.phi.x_bias) * scale.T

    return X_sim.numpy()

# function for simulating continuous-time Koopman model for given initial condition
def forward_sim_cont(model, tvec, init_cond, scale):

    with torch.no_grad():
        t = torch.tensor(tvec.T, device=device)
        z0 = model.phi(torch.tensor((init_cond / scale).T, device=device))
        A = model()
        lam, V = torch.linalg.eig(A)
        Lamt = (lam.repeat(t.shape[0], 1) * t).exp()
        At = V @ (Lamt.diag_embed() @ V.inverse())
        Z_sim = (At.real @ z0.T).squeeze()
        if hasattr(model, 'phi_inv'):
            X_sim = model.phi_inv(Z_sim) * scale.T
        else:
            X_sim = (Z_sim + model.phi.x_bias) * scale.T

    return X_sim.cpu().numpy()


if __name__ == '__main__':

    shapes = ['Angle','CShape','GShape','JShape',
              'LShape','NShape','PShape','RShape',
              'Sshape','WShape','Zshape','BendedLine',
              'DoubleBendedLine','heee','JShape_2','Khamesh',
              'Leaf_1','Leaf_2','Line','Saeghe','Sharpc',
              'Sine','Snake','Spoon','Trapezoid','Worm']

    disc = True         # flag for discrete-time (True) or continuous-time (False) model
    num_trajs = 7       # total number of trajectories in the dataset
    num_demos = 6       # number of trajectories to use for training
    dim_K = 20          # dimensionality of observables phi
    num_tests = 7       # number of leave-one-out cross validations to perform (equal to num of trajs)

    sim_errors = np.zeros((num_tests, len(shapes)))

    # no. of subplots for figure
    nrows = 3 
    ncols = 3

    # iterate over each trajectory for leave-one-out cross validation
    for j in range(num_tests):

        fig, ax = plt.subplots(nrows, ncols, figsize=(6, 6))
        fig.supxlabel('x (mm)') 
        fig.supylabel('y (mm)')

        for n in range(len(shapes)):

            shape = shapes[n]
            if disc: 
                data = spio.loadmat("lasahandwritingdataset/DataDiscrete/" + shape + ".mat")
                dt = data['dt']
            else:
                data = spio.loadmat("lasahandwritingdataset/DataSet/" + shape + ".mat")

            pos, pos_next, vel, vel_next, t = [], [], [], [], []
            traj_len = np.zeros((num_trajs, ), dtype=int)
            for i in range(num_trajs):
                y0 = data['demos'][0, i]['pos'][0][0][:, :-1]
                y1 = data['demos'][0, i]['pos'][0][0][:, 1:]
                len_y = y0.shape[-1]
                pos.append(y0)
                pos_next.append(y1)

                ydot0 = data['demos'][0, i]['vel'][0][0][:, :-1]
                ydot1 = data['demos'][0, i]['vel'][0][0][:, 1:]
                vel.append(ydot0)
                vel_next.append(ydot1)

                traj_len[i] = len_y
                if not disc:
                    t.append(data['demos'][0, i]['t'][0][0])

            test_traj_idx = np.arange(num_trajs)
            test_traj_idx = np.delete(test_traj_idx, j)

            # data matrices
            X = np.vstack([np.hstack([pos[m] for m in test_traj_idx]), np.hstack([vel[m] for m in test_traj_idx])])
            Y = np.vstack([np.hstack([pos_next[m] for m in test_traj_idx]), np.hstack([vel_next[m] for m in test_traj_idx])])

            scale = np.max(np.abs(X), axis=-1, keepdims=True)

            X = X / scale
            Y = Y / scale

            test_traj = j

            init_cond = np.expand_dims(np.hstack([pos[test_traj][:, 0], vel[test_traj][:, 0]]), -1)
            noise = np.vstack([4 * np.random.rand(2, 4) - 2, np.zeros((2, 4))])
            init_conds = init_cond + noise
            num_timesteps = data['demos'][0, test_traj]['pos'][0][0].shape[-1]


            model = torch.load('./lasa_models/' + str(j+1) + '/skel_model_' + shape + '.pt')
            
            # simulate model for test initial condition and random perturbations near it
            X_sim = forward_sim_disc(model, num_timesteps, init_cond, scale)
            X_sims = []
            for i in range(4):
                X_sims.append(forward_sim_disc(model, num_timesteps, np.expand_dims(init_conds[:, i], -1), scale))

            X_test = np.hstack([data['demos'][0, test_traj]['pos'][0][0].T, data['demos'][0, test_traj]['vel'][0][0].T])
            norm_X_test = np.linalg.norm(X_test, ord='fro') ** 2
            sim_errors[j, n] = np.linalg.norm(X_sim - X_test, ord='fro') ** 2 / norm_X_test

            # plotting code
            if n < nrows * ncols:

                ax[int(n / nrows), int(n % nrows)].plot(data['demos'][0, test_traj]['pos'][0][0][0, :], data['demos'][0, test_traj]['pos'][0][0][1, :], 'k', linestyle='-', linewidth=1)
                ax[int(n / nrows), int(n % nrows)].plot(X_sim[:, 0], X_sim[:, 1], 'r', linestyle='--', linewidth=1)
                for i in range(4):
                    ax[int(n / nrows), int(n % nrows)].plot(X_sims[i][:, 0], X_sims[i][:, 1], 'r', linestyle='--', linewidth=1)
                ax[int(n / nrows), int(n % nrows)].plot(init_cond[0, 0], init_cond[1, 0], 'ks', fillstyle='none', markersize=10, label='Start point')
                ax[int(n / nrows), int(n % nrows)].plot(0, 0, 'k*', markersize=10, label='Target')

                x_llim = np.min(X_test[:, 0]) - 8.
                x_ulim = np.max(X_test[:, 0]) + 8.
                y_llim = np.min(X_test[:, 1]) - 8.
                y_ulim = np.max(X_test[:, 1]) + 8.

                ax[int(n / nrows), int(n % nrows)].tick_params(labelsize=6)
                ax[int(n / nrows), int(n % nrows)].set_ylim(y_llim, y_ulim)
                ax[int(n / nrows), int(n % nrows)].set_xlim(x_llim, x_ulim)

        plt.show()

        fig.savefig('lasa_figs/skel_' + str(j+1) + '.pdf', bbox_inches = 'tight', pad_inches=0.01)

    spio.savemat('lasa_sim_errors_loocv.mat', {'sim_errors': sim_errors})

