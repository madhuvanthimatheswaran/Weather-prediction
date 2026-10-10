import pandas as pd
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ====================================================
# CREATE FUTURE TARGETS
# ====================================================

def create_future_targets(df):

    df["rain_today"] = df["rainfall"]
    df["rain_tomorrow"] = df.groupby("station_name")["rainfall"].shift(-1)
    df["rain_nextweek"] = df.groupby("station_name")["rainfall"].shift(-7)
    df["rain_nextmonth"] = df.groupby("station_name")["rainfall"].shift(-30)

    df = df.dropna()

    return df


# ====================================================
# PREPARE FEATURES
# ====================================================

def prepare_features(df):

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
        'rain_lag14',

        'rain_3day_avg',
        'rain_7day_avg',
        'rain_14day_avg'
    ]

    # ================================
    # INPUT FEATURES
    # ================================

    X = df[features].values

    # ================================
    # TARGET VALUES
    # ================================

    y = df[[
        "rain_today",
        "rain_tomorrow",
        "rain_nextweek",
        "rain_nextmonth"
    ]].values


    # ================================
    # SCALING
    # ================================

    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()

    X = scalerX.fit_transform(X)
    y = scalerY.fit_transform(y)


    # ================================
    # SAVE SCALERS
    # ================================

    os.makedirs("models", exist_ok=True)

    joblib.dump(scalerX, "models/scalerX.pkl")
    joblib.dump(scalerY, "models/scalerY.pkl")

    print("Scalers saved to models folder")


    # ================================
    # TRAIN TEST SPLIT
    # ================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, scalerY