import torch
import torch.nn as nn

class CNNLSTM(nn.Module):

    def __init__(self,input_channels=1):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(input_channels,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            32*5*5,
            128,
            batch_first=True
        )

        self.fc = nn.Linear(128,1)

    def forward(self,x):

        batch,t,c,h,w = x.size()

        x = x.view(batch*t,c,h,w)

        x = self.cnn(x)

        x = x.view(batch,t,-1)

        out,_ = self.lstm(x)

        out = self.fc(out[:,-1,:])

        return out