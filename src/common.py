import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score
from scipy.stats import pearsonr

def evaluate_regression(y_train, p_train, y_test, p_test):
    rmse_train = np.sqrt(mean_squared_error(y_train, p_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test, p_test))
    pearson_val = pearsonr(y_test, p_test)[0] if np.std(y_test) > 0 and np.std(p_test) > 0 else np.nan
    return {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "mae_test": mean_absolute_error(y_test, p_test),
        "r2_test": r2_score(y_test, p_test),
        "pearson_test": pearson_val,
        "overfit_gap_rmse": rmse_train - rmse_test
    }

def evaluate_classification(y_train, p_train, y_test, p_test, threshold=0.5):
    auc_train = roc_auc_score(y_train, p_train) if len(np.unique(y_train)) > 1 else np.nan
    auc_test  = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan
    yhat_test = (np.asarray(p_test) >= threshold).astype(int)
    return {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "logloss_test": log_loss(y_test, p_test, labels=[0, 1]),
        "accuracy_test": accuracy_score(y_test, yhat_test),
        "f1_test": f1_score(y_test, yhat_test, pos_label=1),
        "overfit_gap_auc": auc_train - auc_test if auc_train == auc_train and auc_test == auc_test else np.nan
    }

def preprocess_classifier_sae(df, selected_features, thresh_low=0.002, thresh_high=10):
    mask = (df['label'].abs() > thresh_low) & (df['label'].abs() < thresh_high)
    df = df[mask][selected_features].copy()
    df['target'] = (df['label'] > 0).astype(int)

    now = df.index.max()
    max_age = (now - df.index.min()).days
    df['age'] = (now - df.index).days
    df['time_decay'] = 1 - (df['age'] / max_age)
    df['weight'] = df['label'].abs() * df['time_decay'].clip(lower=0)
    return df.dropna()