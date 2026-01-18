import pandas as pd
from src.config.config import DATA_PATH


def get_bitcoin_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df