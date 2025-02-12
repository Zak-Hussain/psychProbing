import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss


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

