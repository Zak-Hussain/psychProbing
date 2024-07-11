import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss


def mcfadden_r2_binary(y_true, y_pred_proba):
    # Compute the log-likelihood of the model
    ll_model = -log_loss(y_true, y_pred_proba, normalize=False)

    # Compute the log-likelihood of the null model (predicting the mean)
    probas_null = np.full_like(y_pred_proba, fill_value=y_true.mean(), dtype=np.float64)
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


def make_binary_scorer():
    return make_scorer(mcfadden_r2_binary, greater_is_better=True, needs_proba=True)

def make_multiclass_scorer():
    return make_scorer(mcfadden_r2_multiclass, greater_is_better=True, needs_proba=True)


def best_logistic_solver(X, dtype):
    """
    Pick the fastest 'l2'-compatible for LogisticCV the given data based on a few heuristics.
    """
    if len(X) < 1000:  # Arbitrary threshold for "small" datasets
        if dtype == 'binary':
            return 'liblinear'
        else:
            return 'lbfgs'
    else:
        return 'saga'


def process_categorical(X, y, outer_cv, inner_cv):
    """Removes classes with too few observations"""
    min_class_n = outer_cv * inner_cv
    classes_to_keep = y.value_counts()[y.value_counts() >= min_class_n].index
    to_keep_bool = y.isin(classes_to_keep)
    X, y = X.loc[to_keep_bool], y.loc[to_keep_bool]
    return X, y


def checker(embed_name, y, dtype, meta, outer_cv, norm_name):
    if embed_name in meta.loc[norm_name, 'associated_embed']:
        return 'associated_embed'
    elif len(y) < 2 * outer_cv:
        return 'too few observations'
    elif (dtype != 'continuous') and (len(y.unique()) < 2):
        return 'too few classes (of sufficient size)'
    else:
        return 'pass'


def k_fold_cross_val(estim, X, y, outer_cv, scoring, n_jobs):
    scores = cross_val_score(estim, X, y, cv=outer_cv, scoring=scoring, n_jobs=n_jobs)
    return scores.mean(), scores.std()