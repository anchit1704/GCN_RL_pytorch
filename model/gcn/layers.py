from gcn.init import *
from gcn.utils import *
import torch
import numpy as np

import torch.nn as nn


def sparse_retain(sparse_matrix, to_retain):
    a_mat = torch.IntTensor([])
    np_indices = np.empty((1, 2), int)  # dtype = np.int32)
    np_values = np.array([])
    
   
    for i in range(len(to_retain)):
        if to_retain[i] == True:
            indices_ = [[int(sparse_matrix.coalesce().indices()[0][i]),  int(sparse_matrix.coalesce().indices()[1][i])]]
            np_indices = np.append(np_indices, indices_, axis = 0)
            
            values_ = int(sparse_matrix.coalesce().values()[i])
            np_values = np.append(np_values, values_)
    np_indices = np.delete(np_indices, 0, axis = 0)
    

    sp_indices = torch.from_numpy(np_indices.T)
    sp_values = torch.from_numpy(np_values)
    
    retain_matrix = torch.sparse_coo_tensor(sp_indices, sp_values, sparse_matrix.shape)

    return retain_matrix

    
def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += torch.FloatTensor(noise_shape).uniform_()
    dropout_mask= (torch.floor(random_tensor).type(torch.BoolTensor))
    pre_out = sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(weight_variable_glorot(input_dim, output_dim)))
        self.dropout = dropout
        self.act = act

    def forward(self, adj, inputs, dropout=0., act=torch.nn.ReLU()):
        x = inputs
        x = torch.nn.Dropout(1-dropout, x)
        x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)
        outputs = act(x)
        return outputs
    
    
    
class GraphConvolutionSparse(nn.Module):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, features_nonzero):
        super(GraphConvolutionSparse, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(weight_variable_glorot(input_dim, output_dim)))
        self.issparse = True
        self.features_nonzero = features_nonzero

    def forward(self, adj, inputs, dropout=0., act=torch.nn.ReLU()):
        x = inputs
        x = dropout_sparse(x, 1-dropout, self.features_nonzero)
        x = torch.spmm(x, self.weight)
        x = torch.spmm(adj, x)
        outputs = act(x)
        return outputs


class RelationalGraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, **kwargs):
        super(RelationalGraphConvolution, self).__init__(**kwargs)
        
        self.weights_1 = nn.Parameter(weight_variable_glorot(input_dim, output_dim, name="weights_1"))
        self.weights_2 = nn.Parameter(weight_variable_glorot(input_dim, output_dim, name="weights_2"))

    def forward(self, inputs, adj_1, adj_2, dropout=0., act=torch.nn.ReLU()):
        adj_1 = convert_coo_to_torch_coo_tensor(adj_1.tocoo())
        adj_2 = convert_coo_to_torch_coo_tensor(adj_2.tocoo())

        m = torch.nn.Dropout(1-dropout)
        x = m(inputs)

        x_1 = torch.mm(x, self.weights_1.type(torch.cuda.FloatTensor))
        x_1 = torch.spmm(adj_1, x_1)

        x_2 = torch.mm(x, self.weights_2.type((torch.cuda.FloatTensor)))
        x_2 = torch.spmm(adj_2, x_2)
        outputs = act(x_1 + x_2)
        return outputs


class RelationalGraphConvolutionSparse(nn.Module):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim,  features_nonzero, **kwargs):
        super(RelationalGraphConvolutionSparse, self).__init__(**kwargs)
        self.weights_1 = nn.Parameter(weight_variable_glorot(input_dim, output_dim, name = "sparse_w1")).to('cuda:0')
        self.weights_2= nn.Parameter(weight_variable_glorot(input_dim, output_dim, name = "sparse_w2")).to('cuda:0')
        self.issparse = True
        self.features_nonzero = features_nonzero


    def forward(self, inputs, adj_1, adj_2,  dropout=0., act=torch.nn.ReLU()):
        adj_1 = convert_coo_to_torch_coo_tensor(adj_1.tocoo())
        adj_2 = convert_coo_to_torch_coo_tensor(adj_2.tocoo())
        x = dropout_sparse(inputs, 1-dropout, self.features_nonzero).to('cuda:0')

        x_1 = torch.spmm(x, self.weights_1).to('cuda:0')
        x_1 = torch.spmm(adj_1, x_1.type(torch.cuda.FloatTensor)).to('cuda:0')

        x_2 = torch.spmm(x, self.weights_2).to('cuda:0')
        x_2 = torch.spmm(adj_2, x_2.type(torch.cuda.FloatTensor)).to('cuda:0')

        outputs = act(x_1 + x_2).to('cuda:0')
        return outputs
