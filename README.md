# skel

Code to reproduce https://arxiv.org/abs/2110.06509

Dependencies:
Pytorch >=1.9.0, Numpy, Scipy, Matplotlib

To train models, run lasa_train.py

To evaluate models (i.e. compute test simulation error and generate figures), run lasa_disc_eval.py

'/lasa_models' contains pre-trained models that can be visualised using lasa_disc_eval.py (used to generate Figure 3 in the paper)

'/lasahandwritingdataset' contains a Matlab script 'discretise.m' to generate discrete-time trajectories from the original dataset (available from https://cs.stanford.edu/people/khansari/download.html)

'/lasahandwritingdataset/DataDiscrete' contains the training data used to train the models for the paper
