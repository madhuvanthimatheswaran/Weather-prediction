import torch
import numpy as np
from src.pinn_model import PINN

def predict(model,X_test,scalerY):

    X_test = torch.tensor(X_test,dtype=torch.float32)

    model.eval()

    pred = model(X_test).detach().numpy()

    pred = scalerY.inverse_transform(pred)

    return pred