import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Flatten arrays
y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()

# ===============================
# 1. Scatter Plot (True vs Predicted)
# ===============================
plt.figure(figsize=(7,6))
plt.scatter(y_true_flat, y_pred_flat, alpha=0.6)
plt.plot([y_true_flat.min(), y_true_flat.max()],
         [y_true_flat.min(), y_true_flat.max()], linestyle="--")
plt.xlabel("True Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("True vs Predicted Rainfall")
plt.grid(True)
plt.show()

# ===============================
# 2. Line Plot Comparison
# ===============================
# Sort values for smoother visualization
indices = np.argsort(y_true_flat)
y_true_sorted = y_true_flat[indices]
y_pred_sorted = y_pred_flat[indices]

plt.figure(figsize=(10,6))
plt.plot(y_true_sorted, label="True Rainfall")
plt.plot(y_pred_sorted, label="Predicted Rainfall")
plt.title("Rainfall Prediction Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Rainfall")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 3. Error Distribution Histogram
# ===============================
errors = y_true_flat - y_pred_flat

plt.figure(figsize=(8,5))
plt.hist(errors, bins=30)
plt.title("Prediction Error Distribution")
plt.xlabel("Error (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ===============================
# 4. Residual Plot
# ===============================
plt.figure(figsize=(8,6))
plt.scatter(y_pred_flat, errors, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Rainfall")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# ===============================
# 5. Performance Metrics
# ===============================
mse = mean_squared_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_flat, y_pred_flat)
r2 = r2_score(y_true_flat, y_pred_flat)

print("Model Performance Metrics")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R² Score:", r2)