import pandas as pd
import os

def read_csvs(results_pth: str) -> pd.DataFrame:
    results = [pd.read_csv(f'{results_pth}{f_name}') for f_name in os.listdir(results_pth)]
    return pd.concat(results, ignore_index=True)
