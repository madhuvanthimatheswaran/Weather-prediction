import sys
import os

# Allow importing from src folder
sys.path.append(os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_time_features, create_lag_features, create_rolling_features
from src.dataset_builder import create_future_targets, prepare_features
from src.train_model import train_model
from src.predict_future import predict


# ====================================================
# LOAD DATA
# ====================================================

print("\nLoading dataset...")

data_path = "data/india_weather_rainfall_data.xlsx"

df = load_and_clean_data(data_path)

# Reduce dataset size for faster training (optional)
df = df.sample(100000, random_state=42)

print("Dataset loaded:", df.shape)


# ====================================================
# FEATURE ENGINEERING
# ====================================================

print("\nCreating features...")

df = create_time_features(df)

df = create_lag_features(df)

df = create_rolling_features(df)

df = create_future_targets(df)


# ====================================================
# DATASET PREPARATION
# ====================================================

print("\nPreparing training data...")

X_train, X_test, y_train, y_test, scalerY = prepare_features(df)

print("Training samples:", X_train.shape)
print("Testing samples :", X_test.shape)


# ====================================================
# TRAIN MODEL
# ====================================================

print("\nTraining model...")

model = train_model(X_train, y_train)


# ====================================================
# MAKE PREDICTIONS
# ====================================================

print("\nPredicting rainfall...")

predictions = predict(model, X_test, scalerY)

y_true = scalerY.inverse_transform(y_test)


# ====================================================
# MODEL PERFORMANCE
# ====================================================

rmse = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
mae = mean_absolute_error(y_true[:, 0], predictions[:, 0])
r2 = r2_score(y_true[:, 0], predictions[:, 0])

print("\n===== MODEL PERFORMANCE =====")

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)


# ====================================================
# SAVE PREDICTIONS
# ====================================================

print("\nSaving prediction results...")

pred_df = pd.DataFrame(predictions, columns=[
    "today_rainfall",
    "tomorrow_rainfall",
    "next_week_rainfall",
    "next_month_rainfall"
])

os.makedirs("results", exist_ok=True)

pred_df.to_csv("results/rainfall_predictions.csv", index=False)

print("Predictions saved to results/rainfall_predictions.csv")


# ====================================================
# PLOT 1: ACTUAL VS PREDICTED
# ====================================================

plt.figure(figsize=(7, 7))

plt.scatter(y_true[:, 0], predictions[:, 0], alpha=0.3)

plt.plot(
    [y_true[:, 0].min(), y_true[:, 0].max()],
    [y_true[:, 0].min(), y_true[:, 0].max()],
    'r--'
)

plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("PINN Rainfall Prediction")

plt.savefig("results/rainfall_prediction_plot.png", dpi=300)

print("Prediction plot saved")

plt.show()


# ====================================================
# PLOT 2: RAINFALL TREND
# ====================================================

plt.figure(figsize=(10, 5))

plt.plot(y_true[:100, 0], label="Actual")
plt.plot(predictions[:100, 0], label="Predicted")

plt.legend()

plt.xlabel("Samples")
plt.ylabel("Rainfall (mm)")
plt.title("Rainfall Prediction Trend")

plt.savefig("results/rainfall_trend_plot.png", dpi=300)

print("Trend plot saved")

plt.show()


# ====================================================
# SAMPLE OUTPUT
# ====================================================

print("\nSample Predictions:")

print(predictions[:10])

print("\nProject execution completed successfully!")