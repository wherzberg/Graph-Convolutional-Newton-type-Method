# Graph Convolutional Newton-type Method
# This script is a demo for trining networks 
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
# Set up some parameters #
#========================#

# Set a default tensor dtype
torch.set_default_tensor_type(torch.DoubleTensor)

# Where are we doing the training?
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("Using device:", device)

# How many iterations of GCNM
iterations = 10

# What does the graph structure look like?
N_samples = 50
N_nodes = 100
N_edges = 500

# What do the models look like?
channels = [250,250,250]

# Set hyperparameters for training and model
model_name = 'sample_models'
split = 0.8
learning_rate = 0.002
batch_size = 10
max_epochs = 10000
patience = 200



#===========================#
# Prepare for the main loop #
#===========================#

# Set up storage tensors
TRUTHS      = torch.zeros((              N_samples, N_nodes))
PREDICTIONS = torch.zeros((1+iterations, N_samples, N_nodes))
UPDATES     = torch.zeros((  iterations, N_samples, N_nodes))
LOSS_TR     = torch.zeros((  iterations, max_epochs))
LOSS_VA     = torch.zeros((  iterations, max_epochs))

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
    
    # Initialize a model and move to device
    model = main.myModel(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model using the dataset
    model, LOSS_TR[k,:], LOSS_VA[k,:] = main.trainModel(model, dataset, optimizer, split, batch_size, max_epochs, patience, start_time)
    
    # Apply the model to the whole dataset
    dataset, PREDICTIONS[k+1,:,:] = main.applyModel(model, dataset)
    
    # Move the model and data back to cpu from device
    model = model.to('cpu')
    for i in range(len(dataset)):
        dataset[i].x = dataset[i].x.to(device)
        dataset[i].y = dataset[i].y.to(device)
        dataset[i].edge_index = dataset[i].edge_index.to(device)
    
    # Finally, save the model
    save_name = 'models/' + model_name + '_' + str(k) + '.pt'
    torch.save(model.state_dict(), save_name)
    print("Saved model as:", save_name)
total_time = time.time() - start_time    



#================#
# Save some info #
#================#

save_name = 'data/' + model_name + '_training_output.mat'
save_data = {
    'model_name'  : model_name,
    'iterations'  : iterations,
    'channels'    : np.array(channels),
    'TRUTHS'      : TRUTHS.numpy(),
    'PREDICTIONS' : PREDICTIONS.numpy(),
    'UPDATES'     : UPDATES.numpy(),
    'LOSS_TR'     : LOSS_TR.numpy(),
    'LOSS_VA'     : LOSS_VA.numpy(),
    'total_time'  : total_time
}
scipy.io.savemat(save_name, save_data)
