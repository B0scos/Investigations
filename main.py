from src.data_load.load import get_bitcoin_data
import numpy as np
import pandas as pd
from src.data_processing.preprocessing import basic_stats

df = get_bitcoin_data()

df = basic_stats(df)



print(df)