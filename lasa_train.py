'''
This script trains SKEL models on the LASA handwriting dataset as described in https://arxiv.org/abs/2110.06509
'''

import numpy as np
import scipy.io as spio
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from koopman_model import KoopmanModel, KoopmanModelCT

from optimizer import SimErrorOptimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    shapes = ['Angle','CShape','GShape','JShape',
              'LShape','NShape','PShape','RShape',
              'Sshape','WShape','Zshape','BendedLine',
              'DoubleBendedLine','heee','JShape_2','Khamesh',
              'Leaf_1','Leaf_2','Line','Saeghe','Sharpc',
              'Sine','Snake','Spoon','Trapezoid','Worm']

    disc = True         # flag for learning discrete-time (True) or continuous-time (False) model
    num_trajs = 7       # total number of trajectories in the dataset
    num_demos = 6       # number of trajectories to use for training
    dim_K = 20          # dimensionality of observables phi
    max_iters = 50000   # number of epochs for training
    lr = 1e-3           # learning rate
    num_tests = 7       # number of leave-one-out cross validations to perform (equal to num of trajs)

    for shape in shapes:

        if disc: 
            data = spio.loadmat("lasahandwritingdataset/DataDiscrete/" + shape + ".mat")
            dt = data['dt']
        else:
            data = spio.loadmat("lasahandwritingdataset/DataSet/" + shape + ".mat")
    
        pos, pos_next, vel, vel_next, t = [], [], [], [], []
        traj_len = np.zeros((num_trajs, ), dtype=int)
        # stack trajectories to produce data matrices with state = [pos; vel]
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

        # data matrices
        X = np.vstack([np.hstack(pos[:num_demos]), np.hstack(vel[:num_demos])])
        Y = np.vstack([np.hstack(pos_next[:num_demos]), np.hstack(vel_next[:num_demos])])

        scale_pos = np.max(np.abs(X), axis=-1, keepdims=True)

        X = X / scale_pos
        Y = Y / scale_pos

        skel_model = KoopmanModel(X.shape[0], dim_K, hidden_dims=[50, 50], recon=True)
        # skel_model = KoopmanModelCT(X.shape[0], dim_K, hidden_dims=[50, 50], recon=True)   

        optimizer = SimErrorOptimizer(skel_model, [X.T, Y.T], [num_demos, demo_len], tvec=tvec, disc=disc, errtype='sim', lr=lr, epochs=max_iters, batch_size=1000, debug=True)
        losses = optimizer.train()
        print('Training model for ' + shape)

        torch.save(skel_model, './lasa_models/skel_' + shape + '.pt')
        spio.savemat('./lasa_models/skel_loss_' + shape + '.mat', {'losses': losses})


    print('Done')
