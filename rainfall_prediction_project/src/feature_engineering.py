import pandas as pd

def create_time_features(df):

    df["year"] = df["date_of_record"].dt.year
    df["month_num"] = df["date_of_record"].dt.month
    df["day"] = df["date_of_record"].dt.day

    return df


def create_lag_features(df):

    df = df.sort_values(["station_name","date_of_record"])

    df["rain_lag1"] = df.groupby("station_name")["rainfall"].shift(1)
    df["rain_lag2"] = df.groupby("station_name")["rainfall"].shift(2)
    df["rain_lag3"] = df.groupby("station_name")["rainfall"].shift(3)
    df["rain_lag7"] = df.groupby("station_name")["rainfall"].shift(7)
    df["rain_lag14"] = df.groupby("station_name")["rainfall"].shift(14)

    return df


def create_rolling_features(df):

    df["rain_3day_avg"] = df.groupby("station_name")["rainfall"].transform(
        lambda x: x.rolling(3).mean()
    )

    df["rain_7day_avg"] = df.groupby("station_name")["rainfall"].transform(
        lambda x: x.rolling(7).mean()
    )

    df["rain_14day_avg"] = df.groupby("station_name")["rainfall"].transform(
        lambda x: x.rolling(14).mean()
    )

    return df