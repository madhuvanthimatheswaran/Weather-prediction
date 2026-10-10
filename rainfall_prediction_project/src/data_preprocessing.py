import pandas as pd
import numpy as np

def load_and_clean_data(path):

    df = pd.read_excel(path)

    df['date_of_record'] = pd.to_datetime(df['date_of_record'])

    # remove extreme rainfall
    df = df[df['rainfall'] < 300]

    # remove duplicates
    df = df.drop_duplicates()

    # drop missing rainfall
    df = df.dropna(subset=['rainfall'])

    # fill numerical
    df.fillna(df.median(numeric_only=True), inplace=True)

    # fill categorical
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df