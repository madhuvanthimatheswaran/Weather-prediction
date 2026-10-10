import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.pinn_model import PINN
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_time_features, create_lag_features, create_rolling_features
from src.dataset_builder import create_future_targets, prepare_features


# ================================
# LOAD DATA
# ================================

data_path = "data/india_weather_rainfall_data.xlsx"

df = load_and_clean_data(data_path)

df = create_time_features(df)
df = create_lag_features(df)
df = create_rolling_features(df)

df = create_future_targets(df)


# ================================
# PREPARE FEATURES
# ================================

X_train, X_test, y_train, y_test, scalerY = prepare_features(df)


# ================================
# LOAD SAVED MODEL
# ================================

model_path = "models/pinn_rainfall_model.pth"

model = PINN(X_train.shape[1])

model.load_state_dict(torch.load(model_path))

model.eval()


# ================================
# PREDICT
# ================================

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

pred = model(X_test_tensor).detach().numpy()

pred = scalerY.inverse_transform(pred)
y_true = scalerY.inverse_transform(y_test)


# remove negative rainfall
pred = np.maximum(pred,0)


# ================================
# METRICS
# ================================

rmse = np.sqrt(mean_squared_error(y_true[:,0], pred[:,0]))
mae = mean_absolute_error(y_true[:,0], pred[:,0])
r2 = r2_score(y_true[:,0], pred[:,0])

print("\n===== MODEL PERFORMANCE =====")

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)


# ================================
# SAVE RESULTS
# ================================

results = pd.DataFrame({
    "Actual_Rainfall": y_true[:,0],
    "Predicted_Rainfall": pred[:,0]
})

results.to_csv("results/model_evaluation.csv", index=False)

print("\nEvaluation saved to results/model_evaluation.csv")


# ================================
# PLOT
# ================================

plt.figure(figsize=(7,7))

plt.scatter(y_true[:,0], pred[:,0], alpha=0.3)

plt.plot(
[y_true[:,0].min(), y_true[:,0].max()],
[y_true[:,0].min(), y_true[:,0].max()],
'r--'
)

plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")

plt.savefig("results/model_accuracy_plot.png")

plt.show()