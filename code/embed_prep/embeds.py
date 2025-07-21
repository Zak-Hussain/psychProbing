import pandas as pd
from tqdm import tqdm
import numpy as np

def fix_corrupt(pulled):
    """Fixes corrupt lines by removing them."""
    lengths = [len(vec) for vec in pulled.values()]
    modal_length = max(set(lengths), key=lengths.count)
    return {word: vec for word, vec in pulled.items() if len(vec) == modal_length}


def pull_txt(file, to_pull=None):
    num_lines = sum(1 for line in file)
    file.seek(0)  # resets file to start
    pulled = {}

    if to_pull:
        for line in tqdm(file, total=num_lines):
            word, *vec = line.split()
            if word in to_pull:
                pulled[word] = vec
    else:
        for line in tqdm(file, total=num_lines):
            word, *vec = line.split()
            pulled[word] = vec

    pulled = fix_corrupt(pulled)
    return pd.DataFrame(pulled).T.astype(float)


def pull_df(df: pd.DataFrame, to_pull: set):
    return df.loc[df.index.isin(to_pull)].astype(float)


def multi_inner_align(dfs: list, drop_na=False) -> list:
    """Aligns multiple dataframes on their index, dropping rows with any NaNs if specified."""
    if drop_na:
        dfs = [arg.dropna() for arg in dfs]
    intersection = sorted(list(set.intersection(*[set(df.index) for df in dfs])))
    return [df.loc[intersection] for df in dfs]


def standardize(df):
    # Standardize
    df = (df - df.mean()) / df.std()
    # Set Nans to zero
    df = df.fillna(0.0)
    return df


def ppmi(df):
    """Computes PPMI of pandas dataframe"""
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    expected = np.outer(df.sum(axis=1), col_totals) / total
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    df[df < 0] = 0.0
    return df