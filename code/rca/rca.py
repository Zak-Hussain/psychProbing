import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm.notebook import tqdm
import pandas as pd


def mcfadden_r2_binary(y_true, y_pred_proba):
    # Ensure y_true is a binary vector
    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true).ravel()

    # Compute the log-likelihood of the model
    ll_model = -log_loss(y_true, y_pred_proba, normalize=False)

    # Compute the log-likelihood of the null model (predicting the mean)
    probas_null = np.full_like(y_pred_proba, fill_value=y_true.mean())
    ll_null = -log_loss(y_true, probas_null, normalize=False)

    # Calculate McFadden's R2
    pseudo_r2 = 1 - (ll_model / ll_null)
    return pseudo_r2


def mcfadden_r2_multiclass(y_true, y_pred_proba):
    # Convert y_true to a binary matrix representation (one-hot encoding)
    lb = LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)

    # Compute the log-likelihood of the model
    ll_model = -log_loss(y_true_binary, y_pred_proba, normalize=False)

    # Compute the log-likelihood of the null model (predicting class proportions)
    class_proportions = y_true_binary.mean(axis=0)
    probas_null = np.array([class_proportions] * len(y_true))
    ll_null = -log_loss(y_true_binary, probas_null, normalize=False)

    # Calculate McFadden's R2
    pseudo_r2 = 1 - (ll_model / ll_null)
    return pseudo_r2


def make_binary_scoring():
    r2 = make_scorer(mcfadden_r2_binary, greater_is_better=True, needs_proba=True)
    return r2


def make_multiclass_scoring():
    r2 = make_scorer(mcfadden_r2_multiclass, greater_is_better=True, needs_proba=True)
    return r2


def best_logistic_solver(X, dtype):
    """Pick the fastest 'l2'-compatible for LogisticCV the given data based on a few heuristics."""
    if len(X) < 1000:  # Arbitrary threshold for "small" datasets
        if dtype == 'binary':
            return 'liblinear'
        else:
            return 'lbfgs'
    else:
        return 'saga'


def process_categorical(outer_cv, inner_cv, X, y):
    """Removes classes with too few observations and returns filtered dataframes"""
    min_class_n = outer_cv * inner_cv
    classes_to_keep = y.value_counts()[y.value_counts() >= min_class_n].index
    to_keep_bool = y.isin(classes_to_keep)
    y_filtered = y.loc[to_keep_bool]
    X_filtered = X.loc[to_keep_bool]

    return X_filtered, y_filtered


def checker(embed_names, y, dtype, associated_embeds, outer_cv):
    """Checks various conditions to determine the status of the data."""
    embed_names = [embed_names] if isinstance(embed_names, str) else embed_names
    associated_embeds = str(associated_embeds).split()

    # Checks for data leakage
    if set(embed_names) & set(associated_embeds):
        return 'associated_embed'

    # Check if there are too few observations
    test_n = int((1 / outer_cv) * len(y))
    if test_n < 20:
        return 'test_n < 20'

    # Check if there are too few classes of sufficient size for non-continuous data
    if dtype != 'continuous' and len(y.unique()) < 2:
        return 'too few classes (of sufficient size)'

    return 'pass'


def run_rca(embeds: dict, norms: pd.DataFrame, norm_meta: pd.DataFrame, n_jobs: int, embed_to_type=None) -> pd.DataFrame:
    # --- Hyperparameters ---
    # Ridge regression
    min_ord, max_ord = -5, 5
    alphas = np.logspace(min_ord, max_ord, max_ord - min_ord + 1)
    ridge = RidgeCV(alphas=alphas)

    # Logistic regression
    Cs = 1 / alphas
    inner_cv = 5
    penalty = 'l2'

    # Cross-validation settings
    outer_cv = 5

    # --- Scorers ---
    binary_scoring = make_binary_scoring()
    multiclass_scoring = make_multiclass_scoring()
    continuous_scoring = 'r2'

    # ---- Results accumulator ---
    results = []

    # --- Main cross-validation loop ---
    for embed_name in tqdm(embeds.keys()):
        embed = embeds[embed_name]

        to_print = []
        for norm_name in tqdm(norms.columns, desc=embed_name):
            # 1. Aligning vocabs
            y = norms[norm_name].dropna()
            X, y = embed.align(y, axis=0, join='inner', copy=True)

            # 2. Determine norm dtype and select estimator
            norm_dtype = norm_meta.loc[norm_name, 'type']

            if norm_dtype in ['binary', 'multiclass']:
                # Process data for classification
                X, y = process_categorical(outer_cv, inner_cv, X, y)

                # Recheck dtype in case processing converted multiclass to binary
                norm_dtype = 'binary' if len(y.unique()) == 2 else 'multiclass'

                solver = best_logistic_solver(X, norm_dtype)
                scoring = binary_scoring if norm_dtype == 'binary' else multiclass_scoring

                estimator = LogisticRegressionCV(
                    Cs=Cs,
                    penalty=penalty,
                    cv=StratifiedKFold(inner_cv),
                    solver=solver,
                    n_jobs=8
                )
            else: # Continuous data
                estimator = ridge
                scoring = continuous_scoring

            # 3. Run cross-validation after final check
            associated_embed = norm_meta.loc[norm_name, 'associated_embed']
            check = checker(embed_name, y, norm_dtype, associated_embed, outer_cv)
            if check == 'pass':
                r2s = cross_val_score(  # stratification is automatically used for classification
                    estimator, X, y,
                    cv=outer_cv, scoring=scoring,
                    n_jobs=n_jobs
                )
                r2_mean, r2_sd = r2s.mean(), r2s.std()
            else:
                r2_mean, r2_sd = np.nan, np.nan

            # 4. Save results
            train_n = int(((outer_cv - 1) / outer_cv) * len(X))
            test_n = len(X) - train_n
            p = X.shape[1]
            embed_type = embed_to_type[embed_name] if embed_to_type else None
            results.append([
                embed_name, embed_type, norm_name, train_n, test_n, p,
                r2_mean, r2_sd, check
            ])

            to_print.append([norm_name, train_n, r2_mean, r2_sd, check])

        # Print top results for completed embedding
        to_print = pd.DataFrame(to_print, columns=['norm', 'train_n', 'r2_mean', 'r2_sd', 'check'])
        print(to_print.sort_values('r2_mean', ascending=False).head(10))

    # Convert final results list to DataFrame
    results = pd.DataFrame(
        results, columns=[
            'embed', 'embed_type', 'norm', 'train_n', 'test_n', 'p',
            'r2_mean', 'r2_sd', 'check'
        ]
    )
    return results



