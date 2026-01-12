from src.data_ingestion.BTC_loader import get_bitcoin_data
from src.wavelets import wavelet_decompose
import numpy as np
import pandas as pd

def main(download_data: bool = True):
    if download_data:
        df = get_bitcoin_data(timeframe="daily", days=3000)
        df.to_csv("data/BTC_data.csv")
    else:
        df = df = pd.read_csv("data/BTC_data.csv", index_col='timestamp', parse_dates=True)

    log_ret = np.log(df['close']).diff().dropna().values

    approximation, details = wavelet_decompose(log_ret, 'haar', 4)
 

if __name__ == "__main__":
    main()
