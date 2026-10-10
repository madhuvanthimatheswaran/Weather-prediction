import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 4)   # today, tomorrow, week, month rainfall
        )

    def forward(self, x):

        return self.net(x)


# ------------------------------------
# Physics Loss Function
# ------------------------------------

def physics_loss(model, x):

    x.requires_grad_(True)

    rain = model(x)[:, 0]   # today's rainfall prediction

    grads = torch.autograd.grad(
        rain,
        x,
        grad_outputs=torch.ones_like(rain),
        create_graph=True
    )[0]

    dR_dlat = grads[:, 0]
    dR_dlon = grads[:, 1]

    wind = x[:, 2]
    temp = x[:, 3]
    pressure = x[:, 6]
    elevation = x[:, 7]

    diffusion = 0.01

    # 1️⃣ Advection–Diffusion (moisture transport)
    adv_diff = wind * (dR_dlat + dR_dlon) - diffusion * (dR_dlat**2 + dR_dlon**2)

    # 2️⃣ Thermodynamic rainfall relation
    thermo = rain - torch.exp(0.1 * temp)

    # 3️⃣ Pressure gradient effect
    pressure_grad = rain + pressure

    # 4️⃣ Orographic rainfall
    orographic = rain - wind * elevation

    loss = (

        torch.mean(adv_diff**2) +
        torch.mean(thermo**2) +
        torch.mean(pressure_grad**2) +
        torch.mean(orographic**2)

    )

    return loss