import torch
import torch.nn as nn
import torch.optim as optim

from src.pinn_model import PINN, physics_loss


def train_model(X_train, y_train):

    # =====================================
    # CONVERT NUMPY TO TENSOR
    # =====================================

    X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # =====================================
    # INITIALIZE MODEL
    # =====================================

    input_dim = X_train.shape[1]

    model = PINN(input_dim)

    # =====================================
    # OPTIMIZER
    # =====================================

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =====================================
    # LOSS FUNCTION
    # =====================================

    criterion = nn.MSELoss()

    # =====================================
    # TRAINING PARAMETERS
    # =====================================

    epochs = 2000

    print("\n===== TRAINING PINN MODEL =====")

    # =====================================
    # TRAINING LOOP
    # =====================================

    for epoch in range(epochs):

        optimizer.zero_grad()

        # Model prediction
        pred = model(X_train)

        # Data loss
        data_loss = criterion(pred, y_train)

        # Physics loss
        phys_loss = physics_loss(model, X_train)

        # Total loss
        loss = data_loss + 0.0001 * phys_loss

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Print training progress
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # =====================================
    # SAVE TRAINED MODEL
    # =====================================

    torch.save(model.state_dict(), "models/pinn_rainfall_model.pth")

    print("\nModel saved to models/pinn_rainfall_model.pth")

    return model