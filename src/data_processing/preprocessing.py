import pandas as pd
import numpy as np
import logging

def basic_stats(df: pd.DataFrame, col = 'close') -> pd.DataFrame:
    df['log_return'] = np.log(df[col]).diff()
    df['rv'] = (df.groupby(df.index.date)['log_return'].transform(lambda x: np.sum(x**2)))
    df['log_rv'] = np.log(df['rv'])
    
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.isna().sum().sum() > 1:
        logging.warning("More than two nan after basic_stats at src/data_processing/process.py")
    
    return df
    