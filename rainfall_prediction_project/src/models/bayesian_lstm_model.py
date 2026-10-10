import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLSTM(nn.Module):

    def __init__(self,input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            128,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(128,1)

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.dropout(out)

        out = self.fc(out)

        return out