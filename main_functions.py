# This script has some functions that are used in main_training and main_testing
#
# Contact: William Herzberg
#          william.herzberg@marquette.edu

import copy
import time
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

def initializeDataset(N_samples, N_nodes, N_edges):
    # A random dataset will be created
    
    # The truth and inputs
    data_true = torch.rand((N_samples, N_nodes))
    data_in   = torch.rand((N_samples, N_nodes))
    
    # Also need to make edges
    edges = torch.randint(N_nodes, (2,N_edges//2))
    edges = torch.cat((edges, torch.flipud(edges)), 1)
    edges = torch.unique(edges, dim=1)
    
    # Now create a list of data objects
    dataset = []
    for i in range(N_samples):
        dataset.append(Data(edge_index = edges, x=data_in[i,:].unsqueeze(dim=1), y=data_true[i,:].unsqueeze(dim=1)))
    
    # Return the dataset
    return dataset


def computeUpdates(dataset):
    # A random update will be simulated for each sample
    
    # For each sample, generate a random update and concatenate with x
    for i in range(len(dataset)):
        dataset[i].x = torch.cat((dataset[i].x, torch.rand_like(dataset[i].x)), dim=1)
    
    # Return the dataset
    return dataset
    
    
class myModel(torch.nn.Module):
    # This class is for the GNN model
    
    def __init__(self, channels):
        super(myModel, self).__init__()
        self.channels = channels
        
        # Make a list of the graph convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        in_channels = 2
        for out_channels in channels:
            self.conv_layers.append(GCNConv(in_channels,out_channels))
            in_channels = out_channels
        self.conv_layers.append(GCNConv(in_channels,1))
        self.reset_parameters()
        
    def reset_parameters(self):
        # A function for resetting the parameters
        for layer in self.conv_layers:
            layer.reset_parameters()
        
    def forward(self, data):
        # The forward pass
        x, ei = data.x, data.edge_index
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x,ei)
            if i < (len(self.conv_layers)-1): # No relu on final layer
                x = torch.nn.functional.relu(x)
        return x



def trainModel(model, dataset, optimizer, split, batch_size, max_epochs, patience, start_time):
    # This is the main training function
        
    # Split the dataset into training and validation and create dataLoaders
    split = int(split*len(dataset))
    dataset_tr = dataset[ :split]
    dataset_va = dataset[split: ]
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size)
    loader_va = DataLoader(dataset_va, batch_size=batch_size)
    
    # Prepare some things
    loss_tr = torch.zeros((max_epochs))
    loss_va = torch.zeros((max_epochs))
    pat = patience
    loss_va_min = 10e6
    
    # Start main training loop
    for epoch in range(max_epochs):
        
        # Do training and validation
        loss_tr[epoch] = train(model, loader_tr, optimizer, len(dataset_tr))
        loss_va[epoch] = valid(model, loader_va,            len(dataset_va))
        
        # Print an update every once in a while
        if (epoch) % 1 == 0:
            elapsed = time.time() - start_time
            hrs = int(elapsed//3600)
            mins = int((elapsed-3600*hrs)//60)
            sec = int((elapsed-3600*hrs-60*mins)//1)
            print('# ({:02d}:{:02d}:{:02d}) Epoch {:3d} | Training Loss {:.4f} | Validation Loss {:.4f}'
                .format( hrs, mins, sec, epoch, loss_tr[epoch]*1000, loss_va[epoch]*1000 ))
                
        # Maybe stop early due to non-decreasing loss_va
        if loss_va[epoch] <= loss_va_min:
            loss_va_min = loss_va[epoch]
            best_model  = copy.deepcopy(model)
            pat = patience
            best_epoch = epoch
        else:
            pat -= 1
            if pat == 0:
                for model_w, best_w in zip(model.parameters(),best_model.parameters()):
                    model_w.data = best_w.data
                print("# Training stopped early on epoch", best_epoch, "and best weights loaded back...")
                print("#=========================================================")
                break
                
    
    return model, loss_tr, loss_va
    
    
def train(model, loader_tr, optimizer, n):
    model.train() # Puts the model in training mode (ex. dropout will happen)

    loss_all = 0
    for data in loader_tr:
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = torch.nn.functional.mse_loss(output, label)
        loss.backward()
        loss_all += data.num_graphs*loss.item()
        optimizer.step()
    return loss_all / n


def valid(model, loader_va, n):
    model.eval() # Puts the model in eval mode (ex. dropout will NOT happen)

    loss_all = 0
    with torch.no_grad():
        for data in loader_va:
            pred = model(data)
            label = data.y
            loss = torch.nn.functional.mse_loss(pred, label)
            loss_all += data.num_graphs*loss.item()
    return loss_all / n


def applyModel(model, dataset):
    # Apply the model to the dataset
    model.eval()
    
    # Initialize storage
    predictions = torch.zeros((len(dataset), dataset[0].x.shape[0]), device=dataset[0].x.get_device())
    
    with torch.no_grad():
        for i in range(len(dataset)):
            predictions[i,:] = model(dataset[i]).squeeze()
            dataset[i].x = predictions[i,:].unsqueeze(dim=1)
    
    return dataset, predictions.to('cpu')
