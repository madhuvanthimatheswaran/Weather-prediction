import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.conv1 = GCNConv(input_dim,64)
        self.conv2 = GCNConv(64,32)

        self.fc = nn.Linear(32,1)

    def forward(self,x,edge_index):

        x = self.conv1(x,edge_index)
        x = torch.relu(x)

        x = self.conv2(x,edge_index)
        x = torch.relu(x)

        x = self.fc(x)

        return x