# import torch
# import torch.nn as nn
# import numpy as np
# import requests
# import joblib
# from datetime import datetime

# from src.pinn_model import PINN


# # ====================================================
# # LOCATION (CHANGE IF NEEDED)
# # ====================================================

# latitude = 10.99
# longitude = 78.68
# elevation = 88


# # ====================================================
# # FETCH WEATHER DATA
# # ====================================================

# print("\nFetching weather data...")

# url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&hourly=pressure_msl"

# response = requests.get(url)
# data = response.json()

# if "current_weather" not in data:
#     print("Weather data not available")
#     exit()

# avg_temp = data["current_weather"]["temperature"]
# wind_speed = data["current_weather"]["windspeed"]

# min_temp = avg_temp - 2
# max_temp = avg_temp + 2

# air_pressure = data["hourly"]["pressure_msl"][0]


# # ====================================================
# # DATE FEATURES
# # ====================================================

# today = datetime.now()

# year = today.year
# month_num = today.month
# day = today.day


# # ====================================================
# # RAIN HISTORY (TEMPORARY VALUES)
# # ====================================================

# rain_lag1 = 0
# rain_lag2 = 0
# rain_lag3 = 0
# rain_lag7 = 0
# rain_lag14 = 0

# rain_3day_avg = 0
# rain_7day_avg = 0
# rain_14day_avg = 0


# # ====================================================
# # CREATE FEATURE VECTOR (19 FEATURES)
# # ====================================================

# features = np.array([[
#     latitude,
#     longitude,
#     wind_speed,
#     avg_temp,
#     min_temp,
#     max_temp,
#     air_pressure,
#     elevation,

#     year,
#     month_num,
#     day,

#     rain_lag1,
#     rain_lag2,
#     rain_lag3,
#     rain_lag7,
#     rain_lag14,

#     rain_3day_avg,
#     rain_7day_avg,
#     rain_14day_avg
# ]])

# print("\nFeature Vector Created")


# # ====================================================
# # LOAD SCALERS
# # ====================================================

# print("Loading scalers...")

# scalerX = joblib.load("models/scalerX.pkl")
# scalerY = joblib.load("models/scalerY.pkl")

# features_scaled = scalerX.transform(features)


# # ====================================================
# # LOAD MODEL
# # ====================================================

# print("Loading trained model...")

# input_dim = features_scaled.shape[1]

# model = PINN(input_dim)

# model.load_state_dict(torch.load("models/pinn_rainfall_model.pth"))

# model.eval()


# # ====================================================
# # PREDICT
# # ====================================================

# features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# prediction = model(features_tensor).detach().numpy()

# prediction = scalerY.inverse_transform(prediction)

# prediction = np.maximum(prediction, 0)


# # ====================================================
# # RESULTS
# # ====================================================

# print("\n==============================")
# print("RAIN FALL PREDICTION SYSTEM")
# print("==============================")

# print("\nToday's Weather:")
# print("Temperature:", avg_temp, "°C")
# print("Wind Speed:", wind_speed, "km/h")
# print("Pressure:", air_pressure, "hPa")

# print("\nPredicted Rainfall:")

# print("Today:", round(prediction[0][0], 2), "mm")
# print("Tomorrow:", round(prediction[0][1], 2), "mm")
# print("Next Week:", round(prediction[0][2], 2), "mm")
# print("Next Month:", round(prediction[0][3], 2), "mm")

# print("\nPrediction Complete")


import requests
import numpy as np
import torch
import torch.nn as nn
import joblib

from src.pinn_model import PINN


# ==========================================
# GET CITY COORDINATES USING GEOCODING API
# ==========================================

def get_city_coordinates(city):

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"

    response = requests.get(url).json()

    if "results" not in response:
        print("City not found")
        exit()

    lat = response["results"][0]["latitude"]
    lon = response["results"][0]["longitude"]

    return lat, lon


# ==========================================
# GET WEATHER DATA
# ==========================================

def get_weather(lat, lon):

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    data = requests.get(url).json()

    weather = data["current_weather"]

    temp = weather["temperature"]
    wind = weather["windspeed"]

    # OpenMeteo does not always return pressure
    pressure = 1013

    return temp, wind, pressure


# ==========================================
# USER INPUT CITY
# ==========================================

print("\n==============================")
print("RAIN FALL PREDICTION SYSTEM")
print("==============================\n")

city = input("Enter Indian City Name: ")

lat, lon = get_city_coordinates(city)

print(f"\nCity Found: {city}")
print(f"Latitude : {lat}")
print(f"Longitude: {lon}")


# ==========================================
# FETCH WEATHER
# ==========================================

print("\nFetching weather data...")

temp, wind, pressure = get_weather(lat, lon)


# ==========================================
# CREATE FEATURE VECTOR (19 FEATURES)
# ==========================================

year = 2026
month = 3
day = 15

features = np.array([[
    lat,
    lon,
    wind,
    temp,
    temp - 3,
    temp + 3,
    pressure,
    100,
    year,
    month,
    day,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
]])

print("\nFeature Vector Created")


# ==========================================
# LOAD SCALERS
# ==========================================

print("Loading scalers...")

scalerX = joblib.load("models/scalerX.pkl")
scalerY = joblib.load("models/scalerY.pkl")

features_scaled = scalerX.transform(features)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

input_dim = features_scaled.shape[1]

model = PINN(input_dim)

model.load_state_dict(torch.load("models/pinn_rainfall_model.pth"))

model.eval()


# ==========================================
# PREDICT
# ==========================================

X = torch.tensor(features_scaled, dtype=torch.float32)

with torch.no_grad():
    prediction = model(X).numpy()

prediction = scalerY.inverse_transform(prediction)


today = prediction[0][0]
tomorrow = prediction[0][1]
week = prediction[0][2]
month = prediction[0][3]


# ==========================================
# RESULTS
# ==========================================

print("\nToday's Weather")
print("Temperature:", temp, "°C")
print("Wind Speed:", wind, "km/h")
print("Pressure:", pressure, "hPa")

print("\nPredicted Rainfall")

print("Today:", round(today,2), "mm")
print("Tomorrow:", round(tomorrow,2), "mm")
print("Next Week:", round(week,2), "mm")
print("Next Month:", round(month,2), "mm")

print("\nPrediction Complete")