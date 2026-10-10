

# # import numpy as np
# # import torch
# # import joblib
# # import requests
# # import matplotlib.pyplot as plt
# # import geopandas as gpd
# # import pandas as pd

# # from src.pinn_model import PINN

# # # ====================================
# # # LOAD MODEL AND SCALERS
# # # ====================================
# # print("Loading trained model...")

# # scalerX = joblib.load("models/scalerX.pkl")
# # scalerY = joblib.load("models/scalerY.pkl")

# # input_dim = scalerX.n_features_in_

# # model = PINN(input_dim)
# # model.load_state_dict(torch.load("models/pinn_rainfall_model.pth"))
# # model.eval()

# # # ====================================
# # # INDIAN STATES (CAPITAL COORDINATES)
# # # ====================================
# # states = [
# #     ("Andhra Pradesh",16.5062,80.6480),
# #     ("Arunachal Pradesh",27.0844,93.6053),
# #     ("Assam",26.1445,91.7362),
# #     ("Bihar",25.5941,85.1376),
# #     ("Chhattisgarh",21.2514,81.6296),
# #     ("Goa",15.4909,73.8278),
# #     ("Gujarat",23.0225,72.5714),
# #     ("Haryana",30.7333,76.7794),
# #     ("Himachal Pradesh",31.1048,77.1734),
# #     ("Jharkhand",23.3441,85.3096),
# #     ("Karnataka",12.9716,77.5946),
# #     ("Kerala",8.5241,76.9366),
# #     ("Madhya Pradesh",23.2599,77.4126),
# #     ("Maharashtra",19.0760,72.8777),
# #     ("Manipur",24.8170,93.9368),
# #     ("Meghalaya",25.5788,91.8933),
# #     ("Mizoram",23.7271,92.7176),
# #     ("Nagaland",25.6751,94.1086),
# #     ("Odisha",20.2961,85.8245),
# #     ("Punjab",30.7333,76.7794),
# #     ("Rajasthan",26.9124,75.7873),
# #     ("Sikkim",27.3314,88.6138),
# #     ("Tamil Nadu",13.0827,80.2707),
# #     ("Telangana",17.3850,78.4867),
# #     ("Tripura",23.8315,91.2868),
# #     ("Uttar Pradesh",26.8467,80.9462),
# #     ("Uttarakhand",30.3165,78.0322),
# #     ("West Bengal",22.5726,88.3639)
# # ]

# # # ====================================
# # # WEATHER FUNCTION
# # # ====================================
# # def get_weather(lat, lon):
# #     url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
# #     data = requests.get(url).json()
# #     weather = data["current_weather"]
# #     temp = weather["temperature"]
# #     wind = weather["windspeed"]
# #     pressure = 1013
# #     return temp, wind, pressure

# # # ====================================
# # # PREDICTION LOOP
# # # ====================================
# # rainfall = []
# # lats = []
# # lons = []
# # names = []

# # print("\nPredicting rainfall for Indian states...\n")

# # for state, lat, lon in states:
# #     temp, wind, pressure = get_weather(lat, lon)

# #     features = np.array([[lat, lon, wind, temp, temp-3, temp+3, pressure,
# #                           100, 2026, 3, 15, 0,0,0,0,0,0,0,0]])

# #     features_scaled = scalerX.transform(features)
# #     X = torch.tensor(features_scaled, dtype=torch.float32)

# #     with torch.no_grad():
# #         pred = model(X).numpy()

# #     pred = scalerY.inverse_transform(pred)

# #     rain_today = max(pred[0][0], 0)  # no negative rainfall

# #     rainfall.append(rain_today)
# #     lats.append(lat)
# #     lons.append(lon)
# #     names.append(state)

# #     print(state, "->", round(rain_today, 2), "mm")

# # # ====================================
# # # CREATE DATAFRAME
# # # ====================================
# # df_map = pd.DataFrame({
# #     "state": names,
# #     "lat": lats,
# #     "lon": lons,
# #     "rainfall": rainfall
# # })

# # # ====================================
# # # LOAD WORLD MAP & FILTER INDIA
# # # ====================================
# # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# # india = world[world['name'] == "India"]

# # # ====================================
# # # CREATE GEODATAFRAME
# # # ====================================
# # gdf = gpd.GeoDataFrame(df_map, geometry=gpd.points_from_xy(df_map.lon, df_map.lat))

# # # ====================================
# # # PLOT INDIA MAP
# # # ====================================
# # fig, ax = plt.subplots(figsize=(10,10))

# # # draw India outline
# # india.boundary.plot(ax=ax, color="black", linewidth=2)

# # # rainfall points
# # gdf.plot(ax=ax, column="rainfall", cmap="Blues", legend=True, markersize=200)

# # # add state names
# # for x, y, label in zip(df_map.lon, df_map.lat, df_map.state):
# #     ax.text(x+0.3, y+0.3, label, fontsize=7)

# # ax.set_title("Predicted Rainfall Map of India (All States)")
# # plt.savefig("results/india_states_rainfall_outline_map.png", dpi=300)
# # plt.show()

# # print("\nMap saved to results/india_states_rainfall_outline_map.png")

import numpy as np
import torch
import joblib
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.pinn_model import PINN

# =========================
# LOAD MODEL AND SCALERS
# =========================

print("Loading trained model...")

scalerX = joblib.load("models/scalerX.pkl")
scalerY = joblib.load("models/scalerY.pkl")

input_dim = scalerX.n_features_in_

model = PINN(input_dim)
model.load_state_dict(torch.load("models/pinn_rainfall_model.pth"))
model.eval()

# =========================
# INDIAN STATES (CAPITAL COORDINATES)
# =========================

states = [
    ("Andhra Pradesh",16.5062,80.6480),
    ("Arunachal Pradesh",27.0844,93.6053),
    ("Assam",26.1445,91.7362),
    ("Bihar",25.5941,85.1376),
    ("Chhattisgarh",21.2514,81.6296),
    ("Goa",15.4909,73.8278),
    ("Gujarat",23.0225,72.5714),
    ("Haryana",30.7333,76.7794),
    ("Himachal Pradesh",31.1048,77.1734),
    ("Jharkhand",23.3441,85.3096),
    ("Karnataka",12.9716,77.5946),
    ("Kerala",8.5241,76.9366),
    ("Madhya Pradesh",23.2599,77.4126),
    ("Maharashtra",19.0760,72.8777),
    ("Manipur",24.8170,93.9368),
    ("Meghalaya",25.5788,91.8933),
    ("Mizoram",23.7271,92.7176),
    ("Nagaland",25.6751,94.1086),
    ("Odisha",20.2961,85.8245),
    ("Punjab",30.7333,76.7794),
    ("Rajasthan",26.9124,75.7873),
    ("Sikkim",27.3314,88.6138),
    ("Tamil Nadu",13.0827,80.2707),
    ("Telangana",17.3850,78.4867),
    ("Tripura",23.8315,91.2868),
    ("Uttar Pradesh",26.8467,80.9462),
    ("Uttarakhand",30.3165,78.0322),
    ("West Bengal",22.5726,88.3639)
]

# =========================
# WEATHER FUNCTION
# =========================

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    data = requests.get(url).json()
    weather = data["current_weather"]
    temp = weather["temperature"]
    wind = weather["windspeed"]
    pressure = 1013
    return temp, wind, pressure

# =========================
# PREDICTION LOOP
# =========================

rainfall = []
lats = []
lons = []
names = []

print("\nPredicting rainfall for Indian states...\n")

for state, lat, lon in states:

    temp, wind, pressure = get_weather(lat, lon)

    features = np.array([[

        lat,
        lon,
        wind,
        temp,
        temp-3,
        temp+3,
        pressure,
        100,  # elevation placeholder
        2026,
        3,
        15,
        0,0,0,0,0,0,0,0  # dummy features
    ]])

    features_scaled = scalerX.transform(features)
    X = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X).numpy()

    pred = scalerY.inverse_transform(pred)

    # Prevent negative rainfall
    rain_today = max(pred[0][0], 0)

    rainfall.append(rain_today)
    lats.append(lat)
    lons.append(lon)
    names.append(state)
    print(state, "->", round(rain_today, 2), "mm")

# =========================
# PLOT INDIA MAP USING CARTOPY
# =========================

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([68, 98, 6, 38])  # India extent

# India borders
ax.add_feature(cfeature.BORDERS.with_scale('50m'))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))

# Rainfall points
scatter = ax.scatter(lons, lats, c=rainfall, cmap="Blues", s=200, transform=ccrs.PlateCarree())
plt.colorbar(scatter, label="Rainfall (mm)")

# Add state names
for x, y, label in zip(lons, lats, names):
    ax.text(x+0.3, y+0.3, label, fontsize=7, transform=ccrs.PlateCarree())

plt.title("Predicted Rainfall Map of India (All States)")

plt.savefig("results/india_states_rainfall_map_cartopy.png", dpi=300)
plt.show()

# import numpy as np
# import torch
# import joblib
# import requests
# import matplotlib.pyplot as plt
# import geopandas as gpd
# import pandas as pd

# from src.pinn_model import PINN

# # ====================================
# # LOAD MODEL AND SCALERS
# # ====================================

# print("Loading trained model...")

# scalerX = joblib.load("models/scalerX.pkl")
# scalerY = joblib.load("models/scalerY.pkl")

# input_dim = scalerX.n_features_in_

# model = PINN(input_dim)
# model.load_state_dict(torch.load("models/pinn_rainfall_model.pth"))
# model.eval()

# # ====================================
# # INDIAN STATES (CAPITAL COORDINATES)
# # ====================================

# states = [
#     ("Andhra Pradesh",16.5062,80.6480),
#     ("Arunachal Pradesh",27.0844,93.6053),
#     ("Assam",26.1445,91.7362),
#     ("Bihar",25.5941,85.1376),
#     ("Chhattisgarh",21.2514,81.6296),
#     ("Goa",15.4909,73.8278),
#     ("Gujarat",23.0225,72.5714),
#     ("Haryana",30.7333,76.7794),
#     ("Himachal Pradesh",31.1048,77.1734),
#     ("Jharkhand",23.3441,85.3096),
#     ("Karnataka",12.9716,77.5946),
#     ("Kerala",8.5241,76.9366),
#     ("Madhya Pradesh",23.2599,77.4126),
#     ("Maharashtra",19.0760,72.8777),
#     ("Manipur",24.8170,93.9368),
#     ("Meghalaya",25.5788,91.8933),
#     ("Mizoram",23.7271,92.7176),
#     ("Nagaland",25.6751,94.1086),
#     ("Odisha",20.2961,85.8245),
#     ("Punjab",30.7333,76.7794),
#     ("Rajasthan",26.9124,75.7873),
#     ("Sikkim",27.3314,88.6138),
#     ("Tamil Nadu",13.0827,80.2707),
#     ("Telangana",17.3850,78.4867),
#     ("Tripura",23.8315,91.2868),
#     ("Uttar Pradesh",26.8467,80.9462),
#     ("Uttarakhand",30.3165,78.0322),
#     ("West Bengal",22.5726,88.3639)
# ]

# # ====================================
# # WEATHER FUNCTION (current)
# # ====================================

# def get_weather(lat, lon):
#     try:
#         url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
#         data = requests.get(url).json()
#         weather = data["current_weather"]
#         temp = weather["temperature"]
#         wind = weather["windspeed"]
#         pressure = 1013  # default
#     except:
#         # fallback if API fails
#         temp, wind, pressure = 25, 5, 1013
#     return temp, wind, pressure

# # ====================================
# # PREDICTION LOOP
# # ====================================

# rainfall = []
# lats = []
# lons = []
# names = []

# print("\nPredicting rainfall for Indian states...\n")

# for state, lat, lon in states:
#     temp, wind, pressure = get_weather(lat, lon)

#     features = np.array([[
#         lat,
#         lon,
#         wind,
#         temp,
#         temp - 3,
#         temp + 3,
#         pressure,
#         100,        # elevation (dummy)
#         2026,       # year
#         3,          # month
#         15,         # day
#         0,0,0,0,0,0,0,0  # lag features placeholders
#     ]])

#     features_scaled = scalerX.transform(features)
#     X = torch.tensor(features_scaled, dtype=torch.float32)

#     with torch.no_grad():
#         pred = model(X).numpy()

#     pred = scalerY.inverse_transform(pred)
#     rain_today = max(pred[0][0], 0)  # prevent negative rainfall

#     rainfall.append(rain_today)
#     lats.append(lat)
#     lons.append(lon)
#     names.append(state)

#     print(state, "->", round(rain_today, 2), "mm")

# # ====================================
# # CREATE DATAFRAME
# # ====================================

# df_map = pd.DataFrame({
#     "state": names,
#     "lat": lats,
#     "lon": lons,
#     "rainfall": rainfall
# })

# # ====================================
# # LOAD INDIA OUTLINE MAP
# # ====================================

# # Path to downloaded shapefile folder
# # You need: .shp, .shx, .dbf, .prj all in this folder
# world = gpd.read_file("maps/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
# india = world[world["ADMIN"] == "India"]

# # ====================================
# # CREATE GEODATAFRAME
# # ====================================

# gdf = gpd.GeoDataFrame(
#     df_map,
#     geometry=gpd.points_from_xy(df_map.lon, df_map.lat)
# )

# # ====================================
# # PLOT INDIA MAP
# # ====================================

# fig, ax = plt.subplots(figsize=(10,10))

# # draw India outline
# india.boundary.plot(ax=ax, color="black", linewidth=2)

# # plot rainfall points
# gdf.plot(
#     ax=ax,
#     column="rainfall",
#     cmap="Blues",
#     legend=True,
#     markersize=200
# )

# # add state names
# for x, y, label in zip(df_map.lon, df_map.lat, df_map.state):
#     ax.text(x + 0.3, y + 0.3, label, fontsize=8)

# ax.set_title("Predicted Rainfall Map of India (All States)")

# plt.savefig("results/india_states_rainfall_outline_map.png", dpi=300)
# plt.show()

# print("\nMap saved to results/india_states_rainfall_outline_map.png")