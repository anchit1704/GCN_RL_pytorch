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
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              features_nonzero=self.features_nonzero
                                                  ))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=FLAGS.hidden2
                                            ))
        
    def forward(self, adj, inputs, act = torch.nn.ReLU(), dropout =0.):
        hidden1 = self.layers[0](adj, inputs, act, dropout)
        hidden2 = self.layers[1](adj, hidden1, act, dropout)
        out = nn.Softmax(hidden2).to('cuda:0')

        return out


class RGCN(nn.Module):
    def __init__(self, num_features, features_nonzero):
       # with tf.variable_scope(scope):
            super(RGCN, self).__init__()
            self.input_dim = num_features
            self.features_nonzero = features_nonzero
            self.layers = nn.ModuleList()
            self.layers.append(RelationalGraphConvolutionSparse(input_dim=self.input_dim,
                                                    output_dim=FLAGS.hidden1,  # replace tf flag
                                                    features_nonzero=self.features_nonzero
                                                    ))
            self.layers.append(RelationalGraphConvolution(input_dim=FLAGS.hidden1,
                                                      output_dim=FLAGS.hidden2
                                                  ))


    def forward(self, inputs, adj_1, adj_2, dropout = 0., act = torch.nn.ReLU()):
        layer1 = self.layers[0]
        hidden1 = self.layers[0](inputs, adj_1, adj_2, dropout, act)
        layer2 = self.layers[1]
        hidden2 = self.layers[1](hidden1, adj_1, adj_2, dropout, act)

        outputs = torch.mean(hidden2, 0, keepdim=True)
        
        return outputs


class RGCN2(nn.Module):
    def __init__(self, num_features, features):
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