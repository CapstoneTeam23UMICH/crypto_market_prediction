import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


class ClassifierSupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.SiLU(), nn.BatchNorm1d(64)
        )
        self.bottleneck = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.SiLU(), nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x, noise_std=0.0):
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        recon = self.decoder(latent)
        out = self.classifier(latent)
        return recon, out, latent

def model_registry_regression(mode="find_best_model", random_state=42):
    """
    Returns: (model_class, param_grid)
    Modes: "find_best_model" | "best_model_single_param"
    """
    if mode == "best_model_single_param":
    ######### Will convert to dynamic one MLflow is in place #################
        lgb_grid = {
            "num_leaves": [30],
            "max_depth": [5],
            "subsample": [0.8],
            "reg_alpha": [0.1],
            "reg_lambda": [0.1],
            "learning_rate": [0.05],
            "n_estimators": [200],
            "objective": ["regression"],
            "random_state": [random_state]
        }
        return {
            "LGBM": (LGBMRegressor, lgb_grid)
        }

    elif mode == "find_best_model":
        lgb_grid = {
            "num_leaves": [30],
            "max_depth": [3, 5, 10],
            "subsample": [0.8],
            "reg_alpha": [0.1, 0.5],
            "reg_lambda": [0.1, 0.5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200],
            "objective": ["regression"],
            "random_state": [random_state]
        }
        xgb_grid = {
            "max_depth": [3, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_alpha": [0.1, 0.5],
            "reg_lambda": [0.1, 0.5],
            "gamma": [0, 1],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200],
            "objective": ["reg:squarederror"],
            "tree_method": ["hist"],
            "random_state": [random_state]
        }
        mlp_grid = {
            "hidden_layer_sizes": [128, 256],
            "alpha": [0.05, 0.08, 0.1],
            "learning_rate_init": [0.001, 0.005, 0.01],
            "activation": ["relu"],
            "solver": ["adam"],
            "max_iter": [200],
            "random_state": [random_state]
        }
        return {
            "LGBM": (LGBMRegressor, lgb_grid),
            "XGB": (XGBRegressor, xgb_grid),
            "MLP": (MLPRegressor, mlp_grid)
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

def model_registry_autoencoder(mode="find_best_model", random_state=42):
    """
    Returns: (model_class, param_grid)
    Modes: "find_best_model" | "best_model_single_param"
    """
    if mode == "best_model_single_param":
        classifier_sae_grid = {
            "latent_dim": [16],
            "lr": [0.001],
            "weight_decay": [0.08],
            "noise_std": [0.01]
        }
        return {"classifier_SAE": (ClassifierSupervisedAutoencoder, classifier_sae_grid)}

    elif mode == "find_best_model":
        classifier_sae_grid = {
            "latent_dim": [8, 16, 32],
            "lr": [0.001, 0.005, 0.01],
            "weight_decay": [0.05, 0.08, 0.1],
            "noise_std": [0.01, 0.05]
        }
        return {"classifier_SAE": (ClassifierSupervisedAutoencoder, classifier_sae_grid)}

    else:
        raise ValueError(f"Unknown mode: {mode}")
