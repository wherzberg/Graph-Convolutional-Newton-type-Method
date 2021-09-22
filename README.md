# Graph-Convolutional-Newton-type-Method
These are suplementary files for the manuscript "Graph Convolutional Networks for Model-Based Learning in Nonlinear Inverse Problems" by William Herzberg, Daniel B. Rowe, Andreas Hauptmann, and Sarah J. Hamilton. This manuscript can be found here: https://arxiv.org/abs/2103.15138.

**main_functions.py**
This file contains functions that are imported and used in the other two files. Note that as they are, the functions `initializeDataset()` and `computeUpdates()` only use randomly simulated numbers as opposed to real data or methods to compute real updates. Anyone wishing to do meaningful work with these files will need to modify these two functions for their own application.

**main_training.py**
This file can be run to train a series of networks. The trained network weights are saved in a subfolder called 'models' and the data used in training is saved in a subfolder called 'data'. The trained networks and parts of the training data will be loaded in when testing the networks.

**main_testing.py**
This file can be run to test a series of previously trained and saved networks. The saved networks are loaded in and applied to new data not used during training. The new test data is saved in a subfolder called 'data'.
