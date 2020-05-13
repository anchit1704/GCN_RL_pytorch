import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GraphConvolution
from gcn.layers import GraphConvolutionSparse
from gcn.layers import RelationalGraphConvolution
from gcn.layers import RelationalGraphConvolutionSparse
from absl import flags

FLAGS = flags.FLAGS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device='cpu'
print('gcn_models:', device)

class GCN(nn.Module):
    def __init__(self, num_features, features_nonzero):
        super(GCN, self).__init__()
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        
    def forward(self, adj):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=adj,
                                              features_nonzero=self.features_nonzero,
                                              act=torch.nn.ReLU,
                                              dropout=self.dropout,
                                              logging=self.logging).to(device)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=FLAGS.hidden2,
                                        adj=adj,
                                        act=lambda x: x,
                                        dropout=self.dropout,
                                        logging=self.logging).to(device)(self.hidden1)

        self.outputs = torch.nn.Softmax(self.hidden2).to(device)

        self._loss()
        self._accuracy()

    def _loss(self):
        self.loss = masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.labels_mask)


class RGCN(nn.Module):
    def __init__(self, num_features, features_nonzero):
       # with tf.variable_scope(scope):
            super(RGCN, self).__init__()

           # self.inputs = features
            #self.adj_1 = adj_1
            #self.adj_2 = adj_2
            #self.dropout = dropout
            self.input_dim = num_features
            self.features_nonzero = features_nonzero

           # self.build()

    def forward(self, adj_1, adj_2, dropout, inputs):
        self.hidden1 = RelationalGraphConvolutionSparse(input_dim=self.input_dim,
                                                        output_dim=FLAGS.hidden1,   #replace tf flag
                                                        adj_1= adj_1,
                                                        adj_2= adj_2,
                                                        features_nonzero=self.features_nonzero,
                                                        act=torch.nn.ReLU(),
                                                        dropout=dropout
                                                        )(inputs)

        self.hidden2 = RelationalGraphConvolution(input_dim=FLAGS.hidden1,
                                                  output_dim=FLAGS.hidden2,
                                                  adj_1=adj_1,
                                                  adj_2=adj_2,
                                                  act=torch.nn.ReLU(),
                                                  dropout=dropout
                                                  )(self.hidden1)

        self.outputs = torch.mean(self.hidden2, 0, keepdim=True)
        
        return self.outputs


class RGCN2(nn.Module):
    def __init__(self,features, adj_1, adj_2, dropout, num_features, scope):
        #with tf.variable_scope(scope):
            super(RGCN2, self).__init__()

            self.inputs = features
            self.adj_1 = adj_1
            self.adj_2 = adj_2
            self.dropout = dropout
            self.input_dim = num_features

           # self.build()

    def forward(self):
        self.hidden1 = RelationalGraphConvolution(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj_1=self.adj_1,
                                                  adj_2=self.adj_2,
                                                  act=torch.nn.ReLU,
                                                  dropout=self.dropout,
                                                  logging=self.logging)(self.inputs)

        self.outputs = RelationalGraphConvolution(input_dim=FLAGS.hidden1,
                                                  output_dim=FLAGS.hidden2,
                                                  adj_1=self.adj_1,
                                                  adj_2=self.adj_2,
                                                  act=torch.nn.ReLU,
                                                  dropout=self.dropout,
                                                  logging=self.logging)(self.hidden1)
        
        return self.outputs