


# ============================================================
# RAINFALL MODEL COMPARISON (FINAL ENHANCED VERSION)
# ANN | LSTM | CNN-LSTM | TRANSFORMER | PINN
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"C:\Users\HP\Documents\Final Year Project\rainfall_prediction_project\data\india_weather_rainfall_data.xlsx"

SAVE_DIR = r"C:\Users\HP\Documents\Final Year Project\rainfall_prediction_project\models_result"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_excel(DATA_PATH)

df["date_of_record"] = pd.to_datetime(df["date_of_record"])

df = df[df["rainfall"] < 300]

df = df.drop_duplicates()

df.fillna(df.median(numeric_only=True), inplace=True)

# 🔥 MEMORY SAFE SAMPLE
df = df.sample(40000, random_state=42)

df = shuffle(df, random_state=42)

print("Dataset:", df.shape)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("Creating features...")

df["year"] = df["date_of_record"].dt.year
df["month_num"] = df["date_of_record"].dt.month
df["day"] = df["date_of_record"].dt.day

df = df.sort_values(["station_name","date_of_record"])

for lag in [1,2,3,7]:

    df[f"rain_lag{lag}"] = df.groupby("station_name")["rainfall"].shift(lag)

df["rain_3day_avg"] = df.groupby("station_name")["rainfall"].transform(
    lambda x: x.rolling(3).mean()
)

df.dropna(inplace=True)

# ============================================================
# TARGET
# ============================================================

df["rain_today"] = df["rainfall"]

# ============================================================
# FEATURES
# ============================================================

features = [

'latitude',
'longitude',
'wind_speed',
'avg_temp',
'min_temp',
'max_temp',
'air_pressure',
'elevation',

'year',
'month_num',
'day',

'rain_lag1',
'rain_lag2',
'rain_lag3',
'rain_lag7',

'rain_3day_avg'

]

X = df[features].values
y = df[["rain_today"]].values

# ============================================================
# SCALING
# ============================================================

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

X = scalerX.fit_transform(X)
y = scalerY.fit_transform(y)

# ============================================================
# SPLIT
# ============================================================

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ============================================================
# SEQUENCE CREATION
# ============================================================

SEQ_LEN = 5

def create_sequences(X,y):

    Xs, ys = [], []

    for i in range(len(X)-SEQ_LEN):

        Xs.append(X[i:i+SEQ_LEN])
        ys.append(y[i+SEQ_LEN])

    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train,y_train)
X_test_seq, y_test_seq = create_sequences(X_test,y_test)

# ============================================================
# MODELS
# ============================================================

class ANN(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,1)

        )

    def forward(self,x):

        return self.net(x)


class LSTMModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            32,
            batch_first=True
        )

        self.fc = nn.Linear(32,1)

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        return self.fc(out)


class CNNLSTM(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.conv = nn.Conv1d(
            input_dim,
            16,
            kernel_size=2
        )

        self.lstm = nn.LSTM(
            16,
            32,
            batch_first=True
        )

        self.fc = nn.Linear(32,1)

    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.conv(x)

        x = x.permute(0,2,1)

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        return self.fc(out)


class TransformerModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.embedding = nn.Linear(input_dim,32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.fc = nn.Linear(32,1)

    def forward(self,x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x[:,-1,:]

        return self.fc(x)


class PINN(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,1)

        )

    def forward(self,x):

        return self.net(x)

# ============================================================
# TRAIN FUNCTION (EARLY STOPPING)
# ============================================================

def train_model(model,X,y):

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(),lr=0.0005)

    criterion = nn.MSELoss()

    batch_size = 64

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X,dtype=torch.float32),
        torch.tensor(y,dtype=torch.float32)
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    losses=[]

    best_loss=float("inf")
    patience=5
    wait=0

    for epoch in range(50):

        model.train()

        total_loss=0

        for Xb,yb in loader:

            Xb=Xb.to(DEVICE)
            yb=yb.to(DEVICE)

            optimizer.zero_grad()

            pred=model(Xb)

            loss=criterion(pred,yb)

            loss.backward()

            optimizer.step()

            total_loss+=loss.item()

        avg_loss=total_loss/len(loader)

        losses.append(avg_loss)

        print(f"Epoch {epoch+1} Loss:{avg_loss:.6f}")

        if avg_loss < best_loss:

            best_loss = avg_loss
            wait = 0

        else:

            wait += 1

            if wait >= patience:

                print("Early stopping triggered")
                break

    return losses

# ============================================================
# EVALUATE
# ============================================================

def evaluate(model,X,y):

    model.eval()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X,dtype=torch.float32),
        torch.tensor(y,dtype=torch.float32)
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False
    )

    preds=[]

    with torch.no_grad():

        for Xb,_ in loader:

            Xb=Xb.to(DEVICE)

            pred=model(Xb)

            preds.append(pred.cpu().numpy())

    pred=np.vstack(preds)

    pred=scalerY.inverse_transform(pred)
    y_true=scalerY.inverse_transform(y)

    rmse=np.sqrt(mean_squared_error(y_true,pred))
    mae=mean_absolute_error(y_true,pred)
    r2=r2_score(y_true,pred)

    return rmse,mae,r2,pred

# ============================================================
# TRAIN ALL MODELS
# ============================================================

results={}
loss_dict={}

models={

"ANN": ANN(X_train.shape[1]),
"LSTM": LSTMModel(X_train_seq.shape[2]),
"CNN-LSTM": CNNLSTM(X_train_seq.shape[2]),
"Transformer": TransformerModel(X_train_seq.shape[2]),
"PINN": PINN(X_train.shape[1])

}

for name,model in models.items():

    print(f"\nTraining {name}...")

    if name in ["ANN","PINN"]:

        losses=train_model(model,X_train,y_train)

        results[name]=evaluate(model,X_test,y_test)

    else:

        losses=train_model(model,X_train_seq,y_train_seq)

        results[name]=evaluate(model,X_test_seq,y_test_seq)

    loss_dict[name]=losses

# ============================================================
# SAVE METRICS
# ============================================================

metrics=[]

for name,val in results.items():

    rmse,mae,r2,_=val

    metrics.append([name,rmse,mae,r2])

metrics_df=pd.DataFrame(
    metrics,
    columns=["Model","RMSE","MAE","R2"]
)

metrics_df.to_csv(
    os.path.join(SAVE_DIR,"model_metrics.csv"),
    index=False
)

print("\nMetrics saved!")

# ============================================================
# VISUAL COMPARISON (IMPORTANT)
# ============================================================

sample_true = scalerY.inverse_transform(y_test[:100])

for name,val in results.items():

    _,_,_,pred = val

    plt.figure()

    plt.plot(sample_true,label="Actual")

    plt.plot(pred[:100],label=name)

    plt.legend()

    plt.title(f"{name} Prediction Comparison")

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"{name}_prediction_plot.png"
        )
    )

    plt.close()

# ============================================================
# COMBINED PLOT
# ============================================================

plt.figure()

plt.plot(sample_true,label="Actual")

for name,val in results.items():

    _,_,_,pred = val

    plt.plot(pred[:100],label=name)

plt.legend()

plt.title("All Model Prediction Comparison")

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "all_models_prediction.png"
    )
)

plt.close()

print("\nAll visualizations saved successfully!")