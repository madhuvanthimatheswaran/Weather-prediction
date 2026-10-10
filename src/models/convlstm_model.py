import torch
import torch.nn as nn

class ConvLSTM(nn.Module):

    def __init__(self,input_channels=1):

        super().__init__()

        self.conv = nn.Conv2d(
            input_channels,
            16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            16*10*10,
            128,
            batch_first=True
        )

        self.fc = nn.Linear(128,1)

    def forward(self,x):

        batch,t,c,h,w = x.size()

        x = x.view(batch*t,c,h,w)

        x = self.relu(self.conv(x))

        x = x.view(batch,t,-1)

        out,_ = self.lstm(x)

        out = self.fc(out[:,-1,:])

        return out