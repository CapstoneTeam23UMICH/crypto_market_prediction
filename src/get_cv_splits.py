"""
Time-series cross-validation splits with optional purge gap before validation.
"""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def get_cv_splits(df,
    n_splits = 5,
    method= "purged",
    purge_gap_days= 7,
    start = None,
    end = None,
    return_positions = False
):
    """
    Generate time-series CV splits, supporting both purged and regular methods.

    Args:
        df (DataFrame): Input dataframe with datetime index.
        n_splits (int): Number of splits.
        method (str): "purged" or "regular".
        purge_gap_days (int): Gap in days to purge before validation (only for purged).
        start (str or Timestamp, optional): Start date to subset data.
        end (str or Timestamp, optional): End date to subset data.
        return_positions (bool): If True, return integer positions; else datetime indices.

    Returns:
        list[tuple]: Train and validation indices for each split.
    """
    df = df.sort_index()

    if start is not None or end is not None:
        df = df.loc[start:end]

    timestamps = df.index
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []

    if method not in {"regular", "purged"}:
        raise ValueError('method must be either "regular" or "purged".')

    for train_pos, val_pos in tscv.split(df):
        if method == "regular":
            tr_pos_final = train_pos
        else:
            val_times = timestamps[val_pos]
            purge_cutoff = val_times.min() - pd.Timedelta(days=purge_gap_days)
            train_times = timestamps[train_pos]
            tr_mask = train_times < purge_cutoff
            tr_pos_final = train_pos[tr_mask]

        if return_positions:
            train_idx = pd.Index(tr_pos_final)
            val_idx = pd.Index(val_pos)
        else:
            train_idx = timestamps[tr_pos_final]
            val_idx = timestamps[val_pos]

        folds.append((train_idx, val_idx))

    return folds

cv_policy = {
    "LGBM": {"method": "regular", "params": {"n_splits": 5}},
    "XGB":  {"method": "regular", "params": {"n_splits": 5}},
    "MLP":  {"method": "regular", "params": {"n_splits": 5}},
    "classifier_SAE":  {"method": "purged", "params": {"n_splits": 5, "purge_gap_days": 7}},
}


def get_folds_for_model(model_key, df, **overrides):
    """
    Fetch folds according to the CV policy for the model key.
    Overrides can set start/end.
    """
    if model_key not in cv_policy:
        raise KeyError(f"No CV policy for model '{model_key}'. Add it to cv_policy.")
    policy = cv_policy[model_key].copy()
    method = overrides.pop("method", policy["method"])
    params = {**policy.get("params", {}), **overrides}
    return get_cv_splits(df, method=method, **params)