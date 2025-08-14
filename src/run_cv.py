"""
Cross-validation and grid search utilities for training, evaluating, and logging models with MLflow.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Callable, Any
from itertools import product
from tqdm import tqdm
import os

import json
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.lightgbm
import mlflow.xgboost

from .get_cv_splits import get_folds_for_model
from .model_registry import (
    model_registry_autoencoder,
    model_registry_regression,
)
from .model_fit_predict import (
    fit_predict_classifier_sae,
    fit_predict_mlp,
    fit_predict_tree,
)
from .common import (
    preprocess_classifier_sae,
    evaluate_regression,
    evaluate_classification,
)

def to_tensor(df, device):
    return torch.tensor(df.to_numpy(), dtype=torch.float32, device=device)


def run_cv(
    model_key,
    df,
    selected_features,
    target = "label",
    mode = "find_best_model",
    params_override = None
):
    """
    Run cross-validation for a given model.

    Args:
        model_key (str): Key identifying the model in the registry.
        df (DataFrame): Input dataset.
        selected_features (list): Features to use for training.
        target (str): Target column name.
        mode (str): Model registry mode.
        params_override (dict, optional): Override default parameters.

    Returns:
        dict: Folds, predictions, metrics, and trained models.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_key == "classifier_SAE":
        registry = model_registry_autoencoder(mode=mode)
        (ModelClass, param_grid), = registry.values()
        fit_fn: Callable = fit_predict_classifier_sae
        df = preprocess_classifier_sae(df, selected_features + ['label'])
    else:
        registry = model_registry_regression(mode=mode)
        if model_key not in registry:
            raise KeyError(
                f"Model key '{model_key}' not found in regression registry."
            )
        ModelClass, param_grid = registry[model_key]
        if model_key == "MLP":
            fit_fn = fit_predict_mlp
        else:
            fit_fn = fit_predict_tree

    if params_override is not None:
        params = params_override
    else:
        params = {
            k: (v[0] if isinstance(v, (list)) else v)
            for k, v in param_grid.items()
        }
    
    cv_folds = get_folds_for_model(model_key, df)

    y_train_preds: List[np.ndarray] = []
    y_val_preds: List[np.ndarray] = []
    metrics: List[Dict[str, float]] = []
    fold_indices: List[Tuple[pd.Index, pd.Index]] = []
    models: List[Any] = []

    for fold_num, (train_idx, val_idx) in enumerate(cv_folds):
        df_train = df.loc[train_idx]
        df_val = df.loc[val_idx]
        if model_key == "classifier_SAE":
            X_train_t = to_tensor(df_train.loc[:, selected_features], device=device)
            W_train_t = to_tensor(df_train.loc[:, ['weight']], device=device)
            y_train_t = to_tensor(df_train.loc[:, ['target']], device=device).view(-1, 1)
            
            X_val_t   = to_tensor(df_val.loc[:, selected_features], device=device)
            W_val_t   = to_tensor(df_val.loc[:, ['weight']], device=device)
            y_val_t   = to_tensor(df_val.loc[:, ['target']], device=device).view(-1, 1)

            
            y_train_pred, y_val_pred, model = fit_fn(
                (ModelClass, param_grid),
                X_train_t,
                y_train_t,
                W_train_t,
                X_val_t,
                y_val_t,
                W_val_t,
                max_epochs=100
            )

            fold_metrics = evaluate_classification(
                y_train=df_train.loc[:, ['target']].to_numpy(),
                p_train=y_train_pred,
                y_test=df_val.loc[:, ['target']].to_numpy(),
                p_test=y_val_pred,
            )
        else:
            X_train_df = df_train[selected_features]
            y_train_df = df_train[target]
            X_val_df = df_val[selected_features]
            y_val_df = df_val[target]

            if model_key == "MLP":
                X_train_t = to_tensor(X_train_df, device=device)
                y_train_t = to_tensor(y_train_df, device=device).view(-1, 1)
                X_val_t = to_tensor(X_val_df, device=device)
                y_val_t = to_tensor(y_val_df, device=device).view(-1, 1)
                h1, h2 = params['hidden_layer_sizes']
                y_train_pred, y_val_pred, model = fit_fn(
                    (ModelClass, params),
                    X_train_t,
                    y_train_t,
                    X_val_t,
                    y_val_t,
                )
            else:
                y_train_pred, y_val_pred, model = fit_fn(
                    (ModelClass, params),
                    X_train_df.to_numpy(),
                    y_train_df.to_numpy(),
                    X_val_df.to_numpy(),
                    y_val_df.to_numpy(),
                )

            fold_metrics = evaluate_regression(
                y_train=y_train_df.to_numpy(),
                p_train=y_train_pred,
                y_test=y_val_df.to_numpy(),
                p_test=y_val_pred,
            )

        y_train_preds.append(y_train_pred)
        y_val_preds.append(y_val_pred)
        metrics.append(fold_metrics)
        fold_indices.append((train_idx, val_idx))
        models.append(model)

    return {
        "folds": fold_indices,
        "y_train_pred": y_train_preds,
        "y_val_pred": y_val_preds,
        "metrics": metrics,
        "models": models,
    }

def expand_grid(param_grid):
    keys = list(param_grid.keys())
    value_lists = [
        v if isinstance(v, (list, tuple)) else [v]
        for v in param_grid.values()
    ]
    for combo in product(*value_lists):
        yield dict(zip(keys, combo))

def run_grid_search(
    model_key,
    df,
    selected_features,
    target = "label",
    mode = "find_best_model",
    tracking_uri = "/kaggle/working/crypto_market_prediction/mlruns",
    experiment_name = 'test_cv_run',
):
    """
    Run grid search with cross-validation and log results to MLflow.

    Args:
        model_key (str): Key identifying the model in the registry.
        df (DataFrame): Input dataset.
        selected_features (list): Features to use for training.
        target (str): Target column name.
        mode (str): Model registry mode.
        tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.

    Returns:
        list: Tuples of (params, CV results) for each run.
    """

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if model_key == "classifier_SAE":
        registry = model_registry_autoencoder(mode=mode)
        ModelClass, param_grid = registry[model_key]
    else:
        registry = model_registry_regression(mode=mode)
        ModelClass, param_grid = registry[model_key]

    param_list = list(expand_grid(param_grid))
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for idx, params in enumerate(tqdm(param_list, desc=f"Running search for {model_key}")):
        run_name = f"{model_key}_grid_{idx}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("run_type", "cv")
            mlflow.log_param("model_key", model_key)
            mlflow.log_param("features", json.dumps(selected_features))
            for k, v in params.items():
                mlflow.log_param(k, v)

            cv_kwargs = dict(
                model_key=model_key,
                df=df,
                selected_features=selected_features,
                target=target,
                mode=mode,
                params_override=params,
            )

            cv_result = run_cv(**cv_kwargs)
            fold_metrics = cv_result.get("metrics", [])
            if fold_metrics:
                keys = fold_metrics[0].keys()
                for metric_name in keys:
                    try:
                        values = [m[metric_name] for m in fold_metrics]
                        mean_val = float(np.mean(values))
                        mlflow.log_metric(f"mean_{metric_name}", mean_val)
                    except Exception:
                        continue

            try:
                trained_models = cv_result.get("models", [])
                if trained_models:
                    model_obj = trained_models[-1]
                    input_example = df[selected_features].head(10)
                    if model_key in ("classifier_SAE", "MLP"):
                        mlflow.pytorch.log_model(model_obj, name="model",
                                          input_example= to_tensor(input_example))
                    elif model_key == "LGBM":
                        mlflow.lightgbm.log_model(model_obj, name="model",
                                          input_example=input_example)  
                    elif model_key == "XGB":
                        mlflow.xgboost.log_model(model_obj, name="model",
                                          input_example=input_example)
            except Exception:
                pass

            results.append((params, cv_result))

    return results