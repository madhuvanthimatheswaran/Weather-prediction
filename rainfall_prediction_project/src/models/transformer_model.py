import torch
import torch.nn as nn

class TransformerModel(nn.Module):

    def __init__(self,input_dim,model_dim=128,n_heads=4):

        super().__init__()

        self.embedding = nn.Linear(input_dim,model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(model_dim,1)

    def forward(self,x):

        x = self.embedding(x)

        out = self.transformer(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return out