import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('rl_models', device)

class DQN(nn.Module):
    def __init__(self, state_size, output_dim):
        super(DQN, self).__init__()

        self.state_size = state_size

        self.qvalues = nn.Sequential(nn.Linear(state_size, 512, bias=True).to(device),
                                     nn.Tanh().to(device),
                                     nn.Linear(512, 256, bias=True).to(device),
                                     nn.Tanh().to(device),
                                     nn.Linear(256, output_dim, bias=True).to(device),
                                     nn.Tanh().to(device)).to(device)

    def forward(self, state):
       qvalues = self.qvalues(state)
       return qvalues