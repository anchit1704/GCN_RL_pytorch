import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state, output_dim):
        super(DQN, self).__init__()

        self.state = state

        self.hidden1 = torch.nn.Linear(state.shape[1], 512, bias=True)
        self.tanh = torch.nn.Tanh()

        self.hidden2 = torch.nn.Linear(512, 256, bias=True)

        self.qvalues = torch.nn.Linear(256, output_dim, bias=True)

       
       
    def forward(self, state):
           h1 = self.hidden1(state)
           h1 = self.tanh(h1)
           h2 = self.hidden2(h1)
           h2 = self.tanh(h2)
           qvalues = self.qvalues(h2)
           qvalues = self.tanh(qvalues)
           return qvalues