import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta


TIMEFRAME_MAP = {
    'daily': '1d',
    'hourly': '1h',
    '4hour': '4h',
    '15min': '15m',
    '5min': '5m',
    '1min': '1m',
    'weekly': '1w'
}

# candles per day for each timeframe (used to convert "days" to candles)
CANDLES_PER_DAY = {
    '1d': 1,
    '1h': 24,
    '4h': 6,
    '15m': 96,
    '5m': 288,
    '1m': 1440,
    '1w': 1/7
}


def get_bitcoin_data(
    symbol="BTC/USDT",
    timeframe="daily",
    days=30,
    verbose=True,
    exchange=None
):
    """
    Fetch OHLCV data from Binance with proper pagination.

    Args:
        symbol (str): Trading pair (default BTC/USDT)
        timeframe (str): human readable timeframe ('daily','hourly','15min', etc)
        days (int): number of days to fetch
        verbose (bool): print info or stay silent
        exchange: optional ccxt exchange object reuse

    Returns:
        pd.DataFrame indexed by timestamp
    """

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Invalid timeframe '{timeframe}'. "
                         f"Valid options: {list(TIMEFRAME_MAP.keys())}")

    tf = TIMEFRAME_MAP[timeframe]

    if exchange is None:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    candles_needed = int(days * CANDLES_PER_DAY[tf])
    if candles_needed <= 0:
        raise ValueError("Days must be > 0")

    all_data = []
    limit = 1000   # Binance max per request
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    while len(all_data) < candles_needed:
        try:
            batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)

            if not batch:
                break

            all_data.extend(batch)

            # move since forward
            since = batch[-1][0] + 1  

            # be polite
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            if verbose:
                print(f"Retrying due to error: {e}")
            time.sleep(1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)

    # remove duplicates if pagination overlapped
    df = df[~df.index.duplicated(keep='first')]

    # trim extra candles if overshot
    df = df.tail(candles_needed)

    if verbose and len(df) > 0:
        print(f"{len(df)} candles fetched for {symbol} ({timeframe})")
        print(f"  From: {df.index[0]}  To: {df.index[-1]}")
        print(f"  Last price: {df['close'].iloc[-1]:.2f}")

    return df


if __name__ == "__main__":
    df = get_bitcoin_data(timeframe="hourly", days=9000)
    df.to_csv("BTC_data_hour.csv")
