import numpy as np
import scipy.io as spio
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from koopman_model import KoopmanModel, KoopmanModelOrthogonal, KoopmanModelOrthogonalCT, KoopmanModelCT

from optimizer import SimErrorOptimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward_sim_disc(model, init_cond, horizon):

    with torch.no_grad():
        tvec = torch.arange(0, int(horizon)).unsqueeze(-1).to(device)
        A = model()
        lam, V = torch.linalg.eig(A)
        Lamt = torch.pow(lam.repeat(tvec.shape[0], 1), tvec)
        At = V @ (Lamt.diag_embed() @ V.inverse())
        Z_sim = (At.real @ init_cond.T).squeeze()
        if hasattr(model, 'phi_inv'):
            X_sim = model.phi_inv(Z_sim)[:, :2]
        else:
            X_sim = Z_sim[:, -4:-2] + model.phi.x_bias[:2]

    return X_sim.cpu().numpy()

def forward_sim_cont(model, init_cond, tvec):

    with torch.no_grad():
        t = torch.tensor(tvec.T, device=device)
        A = model()
        lam, V = torch.linalg.eig(A)
        Lamt = (lam.repeat(t.shape[0], 1) * t).exp()
        At = V @ (Lamt.diag_embed() @ V.inverse())
        Z_sim = (At.real @ init_cond.T).squeeze()
        if hasattr(model, 'phi_inv'):
            X_sim = model.phi_inv(Z_sim)[:, :2]
        else:
            X_sim = Z_sim[:, -4:-2] + model.phi.x_bias[:2]

    return X_sim.cpu().numpy()



if __name__ == '__main__':

    data = spio.loadmat("CShape_ct.mat")
    num_trajs = 7
    num_demos = 6
    dim_K = 20
    max_iters = 5
    lr = 1e-3
    disc = False
    if disc: dt = data['dt_disc']
    
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

    if disc:
        tvec = None
    else:
        tvec = np.hstack(t).T
    demo_len = traj_len[:num_demos]
    pred_horizon = np.min(demo_len)

    X = np.vstack([np.hstack(pos[:num_demos]), np.hstack(vel[:num_demos])])
    Y = np.vstack([np.hstack(pos_next[:num_demos]), np.hstack(vel_next[:num_demos])])

    scale_pos = np.max(np.abs(X), axis=-1, keepdims=True)

    X = X / scale_pos
    Y = Y / scale_pos

    # koopman = KoopmanModel(X.shape[0], dim_K, hidden_dims=[100, 100], recon=True)
    koopman = KoopmanModelCT(X.shape[0], dim_K, hidden_dims=[100, 100], recon=True)   

    optimizer = SimErrorOptimizer(koopman, [X.T, Y.T], [num_demos, demo_len], tvec=tvec, disc=disc, errtype='sim', lr=lr, epochs=max_iters, batch_size=250, debug=True)
    optimizer.train()

    A = koopman()
    A_koop = A.cpu().detach().numpy()


    Trajs, Metrics = [], []

    fig, ax = plt.subplots(1)

    for i in range(num_trajs):
        init_cond = np.expand_dims(np.hstack([pos[i][:, 0], vel[i][:, 0]]), -1)
        z0 = koopman.phi(torch.tensor((init_cond / scale_pos).T, device=device))

        horizon = traj_len[i] + 1

        if disc:
            X_sim = forward_sim_disc(koopman, z0, horizon)
        else:
            X_sim = forward_sim_cont(koopman, z0, t[i])
        X_sim = X_sim * scale_pos[:2, :].T
        Trajs.append(X_sim)

        ax.plot(X_sim[:, 0], X_sim[:, 1], 'k')
        ax.plot(data['demos'][0, i]['pos'][0][0][0, :], data['demos'][0, i]['pos'][0][0][1, :], 'r', linestyle='--')


    test_traj = 6

    test_traj_len = data['demos'][0, test_traj]['pos'][0][0].shape[-1]
    X_test = data['demos'][0, test_traj]['pos'][0][0].T
    norm_X_test = np.linalg.norm(X_test, ord='fro') ** 2
    sim_error = np.linalg.norm(Trajs[test_traj] - X_test, ord='fro') ** 2 / norm_X_test
    print('Test simulation error', sim_error)

    plt.show()

    print('Done')
