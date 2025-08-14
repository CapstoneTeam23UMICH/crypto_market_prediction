"""
Model training and prediction utilities for SAE, MLP, and tree-based models.
"""

import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import torch.optim as optim
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score

def fit_predict_classifier_sae(
    model_with_param,
    X_train, 
    y_train, 
    w_train,
    X_val,   
    y_val,   
    w_val,
    max_epochs=100
):
    """
    Train and evaluate a supervised autoencoder classifier.

    Args:
        model_with_param (tuple): (Model class, params dict).
        X_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        w_train (Tensor): Training sample weights.
        X_val (Tensor): Validation features.
        y_val (Tensor): Validation labels.
        w_val (Tensor): Validation sample weights.
        max_epochs (int): Number of training epochs.

    Returns:
        tuple: (train predictions, validation predictions, trained model).
    """
    ModelClass, params = model_with_param
    params = {k: (v[0] if isinstance(v, (list)) else v) for k, v in params.items()}
    torch.manual_seed(params.get('random_state', 42))

    input_dim = X_train.shape[1]
    model = ModelClass(input_dim=input_dim, latent_dim=params.get('latent_dim', 32))
    model = model.to(X_train.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.get('lr', 1e-3),
        weight_decay=params.get('weight_decay', 0.0)
    )

    mse_loss  = nn.MSELoss()
    bce_loss  = nn.BCELoss(reduction='none')
    recon_a   = params.get('recon_alpha', 0.1)
    noise_std = params.get('noise_std', 0.0)

    best_auc   = -np.inf
    best_state = deepcopy(model.state_dict())

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        recon, pred, _ = model(X_train, noise_std=noise_std)
        loss_recon = mse_loss(recon, X_train)
        loss_pred = (bce_loss(pred, y_train) * w_train).mean()
        loss = recon_a * loss_recon + (1.0 - recon_a) * loss_pred
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, y_val_pred, _ = model(X_val, noise_std=0.0)
            yv = y_val.detach().cpu().numpy().ravel()
            pv = y_val_pred.detach().cpu().numpy().ravel()
            wv = w_val.detach().cpu().numpy().ravel()
            try:
                val_auc = roc_auc_score(yv, pv, sample_weight=wv)
            except Exception:
                val_auc = -np.inf

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, y_train_pred, _ = model(X_train, noise_std=0.0)
        _, y_val_pred,   _ = model(X_val,   noise_std=0.0)

    y_train_pred = y_train_pred.detach().cpu().numpy().ravel()
    y_val_pred   = y_val_pred.detach().cpu().numpy().ravel()
    return y_train_pred, y_val_pred, model


def fit_predict_mlp(
    model_with_param,
    X_train, 
    y_train,
    X_val,   
    y_val,
    max_epochs = 100,
):
    """
    Train and evaluate a multi-layer perceptron.

    Args:
        model_with_param (tuple): (Model class, params dict).
        X_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        X_val (Tensor): Validation features.
        y_val (Tensor): Validation labels.
        max_epochs (int): Number of training epochs.

    Returns:
        tuple: (train predictions, validation predictions, trained model).
    """
    ModelClass, params = model_with_param
    params = {k: (v[0] if isinstance(v, (list)) else v) for k, v in params.items()}
    torch.manual_seed(params.get('random_state', 42))

    input_dim = X_train.shape[1]
    model = ModelClass(
        input_dim=input_dim,
        hidden_layer_sizes = params.get("hidden_layer_sizes", (128, 256)),
        dropout=params.get("dropout", 0.0),
    ).to(X_train.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 0.0),
    )

    mse_loss = nn.MSELoss(reduction="mean")
    best_mse = float("inf")
    best_state = deepcopy(model.state_dict())

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            mse = torch.mean((y_val_pred - y_val) ** 2).item()

        if mse < best_mse:
            best_mse = mse
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train).detach().cpu().numpy().ravel()
        y_val_pred   = model(X_val).detach().cpu().numpy().ravel()

    return y_train_pred, y_val_pred, model

def fit_predict_tree(
    model_with_param,
    X_train, 
    y_train,
    X_val,   
    y_val,
    eval_metric="rmse",
    verbose=False
):
    """
    Train and evaluate a tree-based model (LightGBM or XGBoost).

    Args:
        model_with_param (tuple): (Model class, params dict).
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        eval_metric (str): Evaluation metric for tree models.
        verbose (bool): Verbosity flag for training.

    Returns:
        tuple: (train predictions, validation predictions, trained model).
    """
    ModelClass, params = model_with_param
    params = {k: (v[0] if isinstance(v, (list)) else v) for k, v in params.items()}
    model = ModelClass(**params)

    if isinstance(model, LGBMRegressor):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred   = model.predict(X_val)
        return y_train_pred, y_val_pred, model

    elif isinstance(model, XGBRegressor):
        model.fit(X_train, y_train, verbose=verbose)
        y_train_pred = model.predict(X_train)
        y_val_pred   = model.predict(X_val)
        return y_train_pred, y_val_pred, model

    else:
        raise TypeError("Got unsupported model")