from src.data_ingestion.BTC_loader import get_bitcoin_data
from src.wavelets import wavelet_decompose

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", context="talk")


def main(download_data: bool = True):
    if download_data:
        df = get_bitcoin_data(timeframe="daily", days=3000)
        df.to_csv("data/BTC_data.csv")
    else:
        df = pd.read_csv(
            "data/BTC_data.csv",
            index_col="timestamp",
            parse_dates=True
        )

    log_ret = np.log(df["close"]).diff().dropna().values

    approximation, details = wavelet_decompose(
        log_ret,
        wavelet="haar",
        level=4
    )

    # =========================
    # Plot wavelet decomposition
    # =========================
    n_plots = 1 + len(details)
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(16, 3 * n_plots),
        sharex=True
    )

    # Approximation
    sns.lineplot(
        x=np.arange(len(approximation)),
        y=approximation,
        ax=axes[0]
    )
    axes[0].set_title("Wavelet Approximation (A4)")

    # Detail coefficients
    for i, detail in enumerate(details, start=1):
        sns.lineplot(
            x=np.arange(len(detail)),
            y=detail,
            ax=axes[i]
        )
        axes[i].set_title(f"Wavelet Detail (D{len(details) - i + 1})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
