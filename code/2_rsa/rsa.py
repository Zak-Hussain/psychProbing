import pandas as pd
import numpy as np
from scipy.stats import rankdata


def compute_rsm(embed: pd.DataFrame) -> pd.DataFrame:
    """Compute representational similarity matrix (RSM) from an embedding matrix."""
    l2_norms = np.linalg.norm(embed, axis=1).reshape(-1, 1)
    embed /= np.where(l2_norms == 0, np.finfo(float).eps, l2_norms)
    return embed @ embed.T


def paired_nan_drop(m_i: np.array, m_j: np.array) -> np.array:
    """Takes 1D arrays. Drops all indices where at least one array contains NaN"""
    nan_bool = np.isnan(m_i) | np.isnan(m_j)
    return m_i[~nan_bool], m_j[~nan_bool]


def triangle_flat(m: np.array) -> np.array:
    """Returns upper triangle (excluding diagonal)"""
    return m[np.triu_indices(len(m), k=1)]


def custom_spearmanr(x: np.array, y: np.array):
    """Appears to use less RAM than scipy.stats.spearmanr. Not sure if it's faster."""
    # Rank the data
    ranked_1 = rankdata(x)
    ranked_2 = rankdata(y)

    # Compute spearman correlation
    return np.corrcoef(ranked_1, ranked_2)[0, 1]

def lower_tri_spearman(rsm_i, rsm_j) -> float:
    rsm_i, rsm_j = triangle_flat(rsm_i), triangle_flat(rsm_j)
    rsm_i, rsm_j = paired_nan_drop(rsm_i, rsm_j)
    corr = custom_spearmanr(rsm_i, rsm_j)
    return corr


def compute_rsa(rsm_i: pd.DataFrame, rsm_j: pd.DataFrame, max_n: int) -> tuple:
    """Returns Spearman correlation between two RSMs"""

    # Aligning RSMs
    voc_i, voc_j = rsm_i.index, rsm_j.index
    intersect = voc_i.intersection(voc_j)
    n_words = len(intersect)
    if n_words > max_n:
        intersect = pd.Series(intersect).sample(max_n, random_state=42)
    rsm_i = rsm_i.loc[intersect, intersect]
    rsm_j = rsm_j.loc[intersect, intersect]

    # Converting to numpy
    rsm_i = rsm_i.to_numpy(copy=False)
    rsm_j = rsm_j.to_numpy(copy=False)

    # Filling self-correlations with nan
    np.fill_diagonal(rsm_i, np.nan), np.fill_diagonal(rsm_j, np.nan)

    corr = lower_tri_spearman(rsm_i, rsm_j)
    return corr, n_words
