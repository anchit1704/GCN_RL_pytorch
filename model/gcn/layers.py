from gcn.init import *
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def sparse_retain(sparse_matrix, to_retain):
    # if sparse_matrix.shape[0] != to_retain.shape[0]:
    if len(sparse_matrix.coalesce().values()) != to_retain.shape[0]:
        raise ValueError("Shape Not Matched!") 
    
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
    dropout_mask= torch.floor(random_tensor.type(torch.BoolTensor))
    #dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class GraphConvolution(Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=torch.nn.ReLU, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.weight = Parameter(weight_variable_glorot(input_dim, output_dim))
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def forward(self, inputs):
        x = inputs
        x = torch.nn.Dropout(1-self.dropout, x)
        x = torch.mm(x, self.weight)
        x = torch.spmm(self.adj, x)    ####--- pytorch equivalent
        outputs = self.act(x)
        return outputs
    
    
    
class GraphConvolutionSparse(Module):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
       # with tf.variable_scope(self.name + '_vars'):
        self.weight = Parameter(weight_variable_glorot(input_dim, output_dim))
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def forward(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = torch.spmm(x, self.weight)    ##### --- pytorch equivalent
        x = torch.spmm(self.adj, x)                ##### --- pytorch equivalent 
        outputs = self.act(x)  
        return outputs


class RelationalGraphConvolution(Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_1, adj_2, dropout=0., act=tf.nn.relu, **kwargs):
        super(RelationalGraphConvolution, self).__init__(**kwargs)
        
        self.weights_1 = Parameter(weight_variable_glorot(input_dim, output_dim, name="weights_1"))
        self.weights_2 = Parameter(weight_variable_glorot(input_dim, output_dim, name="weights_2"))
        self.dropout = dropout
        self.adj_1 = adj_1
        self.adj_2 = adj_2
        self.act = act

    def forward(self, inputs):
        x = torch.nn.dropout(1-self.dropout, inputs)

        x_1 = torch.mm(x, self.weights_1)
        #x_1 = tf.sparse_tensor_dense_matmul(self.adj_1, x_1)
        x_1 = torch.spmm(self.adj_1, x_1)
        # x_1 = tf.matmul(self.adj_1, x_1)

        x_2 = torch.mm(x, self.weights_2)
        x_2 = torch.spmm(self.adj_2, x_2)
        # x_2 = tf.matmul(self.adj_2, x_2)
        outputs = self.act(x_1 + x_2)
        return outputs


class RelationalGraphConvolutionSparse(Module):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_1, adj_2, features_nonzero, dropout=0., act=torch.nn.ReLU, **kwargs):
        super(RelationalGraphConvolutionSparse, self).__init__(**kwargs)
       # with tf.variable_scope(self.name + '_vars'):
        self.weights_1 = Parameter(weight_variable_glorot(input_dim, output_dim))
        self.weights_2= Parameter(weight_variable_glorot(input_dim, output_dim))
        self.dropout = dropout
        self.adj_1 = adj_1
        self.adj_2 = adj_2
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def forward(self, inputs):
        x = dropout_sparse(inputs, 1-self.dropout, self.features_nonzero)

        x_1 = torch.spmm(x, self.weights_1)
        x_1 = torch.spmm(self.adj_1, x_1)

        x_2 = torch.spmm(x, self.weights_2)
        x_2 = torch.spmm(self.adj_2, x_2)

        outputs = self.act(x_1 + x_2)
        return outputs
