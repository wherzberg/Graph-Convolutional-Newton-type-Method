# Graph Convolutional Newton-type Method
# This script is a demo for testing trained networks 
# Instead of using real data and the forward/inverse solvers,
# only simulated random data is used!!!!!!!!!
#
# Contact: William Herzberg
#          william.herzberg@marquette.edu

#=================#
# Import packages #
#=================#

import time
import numpy as np
import torch
import scipy.io

import main_functions as main






#========================#
# Predict using new data #
#========================#

# Where are we doing the testing?
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("Using device:", device)

# What is the test name?
test_name = 'test1'

# What does the graph structure look like?
N_samples = 20
N_nodes = 70
N_edges = 350

# Load some info from training output
model_name = 'sample_models'
load_name = 'data/' + model_name + '_training_output.mat'
training_data = scipy.io.loadmat(load_name)
iterations = training_data['iterations'].item()
channels   = training_data['channels'].squeeze().tolist()
model = main.myModel(channels).to(device)



#===========================#
# Prepare for the main loop #
#===========================#

# Set up storage tensors
TRUTHS      = torch.zeros((              N_samples, N_nodes))
PREDICTIONS = torch.zeros((1+iterations, N_samples, N_nodes))
UPDATES     = torch.zeros((  iterations, N_samples, N_nodes))

# Initialize a random dataset
dataset = main.initializeDataset(N_samples, N_nodes, N_edges)    #<---- Load in a real dataset here instead of using random numbers
for i in range(N_samples):
    TRUTHS[i,:] = dataset[i].y.squeeze()
    PREDICTIONS[0,i,:] = dataset[i].x.squeeze()



#=================#
# Start main loop #
#=================#

start_time = time.time()
for k in range(iterations):
    print("Starting iteration", str(k))
    
    # Simulate random updates for each sample in the dataset
    dataset = main.computeUpdates(dataset)    #<---- Compute real updates using a classical method here instead of using random numbers
    for i in range(N_samples):
        UPDATES[k,i,:] = dataset[i].x[:,1]
    
    # Move the dataset to the device
    for i in range(len(dataset)):
        dataset[i].x = dataset[i].x.to(device)
        dataset[i].y = dataset[i].y.to(device)
        dataset[i].edge_index = dataset[i].edge_index.to(device)
    
    # load in trained parameters for the model and move to device
    load_name = 'models/' + model_name + '_' + str(k) + '.pt'
    model.load_state_dict( torch.load( load_name ) )
    model = model.to(device)

    # Apply the model to the whole dataset
    dataset, PREDICTIONS[k+1,:,:] = main.applyModel(model, dataset)
    
    # Move the model and data back to cpu from device
    model = model.to('cpu')
    for i in range(len(dataset)):
        dataset[i].x = dataset[i].x.to(device)
        dataset[i].y = dataset[i].y.to(device)
        dataset[i].edge_index = dataset[i].edge_index.to(device)
    print("Applied the model:", load_name)
total_time = time.time() - start_time



#================#
# Save some info #
#================#

save_name = 'data/' + model_name + '_' + test_name + '.mat'
save_data = {
    'test_name'   : test_name,
    'model_name'  : model_name,
    'iterations'  : iterations,
    'channels'    : np.array(channels),
    'TRUTHS'      : TRUTHS.numpy(),
    'PREDICTIONS' : PREDICTIONS.numpy(),
    'UPDATES'     : UPDATES.numpy(),
    'total_time'  : total_time
}
scipy.io.savemat(save_name, save_data)
