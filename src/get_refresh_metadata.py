"""
Pull or Refresh Metadata Dataframes
"""

import os
import sys
from datetime import datetime

import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import shap


def maybe_refresh_and_push(
    df_compute_fn,
    filename,
    commit_msg,
    github_token,
    refresh_repo_file,
    github_user="nikolozjaghiashvili",
    github_email="nikoloz.jaghiashvili@gmail.com",
    repo_name="crypto_market_prediction",
    subdir="src/data/output/"
):
    """
    Generic loader/pusher for computed dataframes.
    """
    repo_path = f'/kaggle/working/{repo_name}'
    if not os.path.exists(repo_path):
        print("Cloning GitHub repo")
        os.system(f"git clone https://github.com/CapstoneTeam23UMICH/{repo_name}.git {repo_path}")

    parquet_file_path = os.path.join(repo_path, subdir, filename)

    if not refresh_repo_file and os.path.exists(parquet_file_path):
        print(f"Loading existing {filename}")
        return pd.read_parquet(parquet_file_path)

    df_result = df_compute_fn()
    unix_suffix = int(datetime.utcnow().timestamp())

    sys.path.append(repo_path)
    from src.github_push_file import push_parquet_to_github  # local import to avoid side effects on import

    push_parquet_to_github(
        df=df_result,
        filename=filename,
        commit_msg=commit_msg,
        github_user=github_user,
        github_email=github_email,
        github_token=github_token,
        target_subdir=subdir,
        branch=f"feature/{filename.replace('.parquet','')}_{unix_suffix}"
    )

    return df_result


def get_feature_drift_df(df_train='default', df_test='default', feature_list='default',
                         github_token='default', refresh_repo_file=False):
    """
    Computes and pushes feature drift dataframe using KS test.
    """
    def compute_drift_df():
        drift_scores = {}
        for col in tqdm(feature_list, desc="Computing drift scores"):
            ks = ks_2samp(df_train[col], df_test[col])
            signed_stat = ks.statistic if df_test[col].mean() > df_train[col].mean() else -ks.statistic
            drift_scores[col] = signed_stat
        drift_scores = dict(sorted(drift_scores.items(), key=lambda item: -abs(item[1])))
        return pd.DataFrame([
            {'feature': k, 'ks_signed': v, 'ks_abs': abs(v)}
            for k, v in drift_scores.items()
        ])

    return maybe_refresh_and_push(
        df_compute_fn=compute_drift_df,
        filename="df_feature_drift.parquet",
        commit_msg="Add feature drift dataframe",
        github_token=github_token,
        refresh_repo_file=refresh_repo_file
    )

def get_correlation_train_df(df_train='default', feature_list='default',
                            github_token='default', refresh_repo_file=False):
    """
    Computes and pushes correlation matrix for training data.
    """
    def compute_corr_df():
        corr = df_train[feature_list].corr(method='pearson')
        corr_long = corr.stack().reset_index()
        corr_long.columns = ['x', 'y', 'corr']
        corr_long = corr_long[corr_long['x'] != corr_long['y']]
        corr_long['x_min'] = corr_long[['x', 'y']].min(axis=1)
        corr_long['x_max'] = corr_long[['x', 'y']].max(axis=1)
        corr_long = corr_long.drop_duplicates(subset=['x_min', 'x_max'])
        corr_long['x'] = corr_long['x_min']
        corr_long['y'] = corr_long['x_max']
        return corr_long.drop(columns=['x_min', 'x_max'])

    return maybe_refresh_and_push(
        df_compute_fn=compute_corr_df,
        filename="corr_long_train.parquet",
        commit_msg="Add correlation matrix dataframe",
        github_token=github_token,
        refresh_repo_file=refresh_repo_file
    )


def get_correlation_test_df(df_test='default', feature_list='default',
                           github_token='default', refresh_repo_file=False):
    """
    Computes and pushes correlation matrix for test data.
    """
    def compute_corr_df():
        corr = df_test[feature_list].corr(method='pearson')
        corr_long = corr.stack().reset_index()
        corr_long.columns = ['x', 'y', 'corr']
        corr_long = corr_long[corr_long['x'] != corr_long['y']]
        corr_long['x_min'] = corr_long[['x', 'y']].min(axis=1)
        corr_long['x_max'] = corr_long[['x', 'y']].max(axis=1)
        corr_long = corr_long.drop_duplicates(subset=['x_min', 'x_max'])
        corr_long['x'] = corr_long['x_min']
        corr_long['y'] = corr_long['x_max']
        return corr_long.drop(columns=['x_min', 'x_max'])

    return maybe_refresh_and_push(
        df_compute_fn=compute_corr_df,
        filename="corr_long_test.parquet",
        commit_msg="Add correlation matrix dataframe for test set",
        github_token=github_token,
        refresh_repo_file=refresh_repo_file
    )


def get_autocorrelation_train_df(df='default', feature_list='default',
                                 max_lag=20, github_token='default', refresh_repo_file=False):
    """
    Computes and pushes autocorrelation dataframe for training data.
    """
    def compute_autocorrelation_train_df():
        results = []
        for col in tqdm(feature_list, desc="Computing autocorrelation"):
            try:
                acf_values = acf(df[col].dropna(), nlags=max_lag, fft=True)
                for lag in range(1, max_lag + 1):
                    results.append({
                        'feature': col,
                        'lag': lag,
                        'autocorr': acf_values[lag]
                    })
            except Exception as e:
                print(f"Failed on feature {col}: {e}")
        return pd.DataFrame(results)

    return maybe_refresh_and_push(
        df_compute_fn=compute_autocorrelation_train_df,
        filename="autocorr_long_train.parquet",
        commit_msg="Add autocorrelation dataframe",
        github_token=github_token,
        refresh_repo_file=refresh_repo_file
    )


def get_constant_feature_df(df_train='default', df_test='default', feature_list='default',
                             github_token='default', refresh_repo_file=False):
    """
    Computes and pushes constant feature dataframe for training data.
    """
    def compute_constant_feature_df():
        vc_train = {}
        for col in feature_list:
            vc_train[col] = len(df_train[col].value_counts())

        vc_test = {}
        for col in feature_list:
            vc_test[col] = len(df_test[col].value_counts())

        constant_df = pd.DataFrame({
            'feature': list(vc_train.keys()),
            'constant_in_train': [v == 1 for v in vc_train.values()],
            'constant_in_test': [vc_test[k] == 1 for k in vc_train.keys()]
        })
        return constant_df

    return maybe_refresh_and_push(
        df_compute_fn=compute_constant_feature_df,
        filename="df_constant_features.parquet",
        commit_msg="Add constant features dataframe",
        github_token=github_token,
        refresh_repo_file=refresh_repo_file
    )

def get_adversarial_validation_df(df_train='default', df_test='default', feature_list='default',
                                 max_iter=8, random_state = 42, threshold = 0.75, github_token='default', 
                                 refresh_repo_file=False):
    """
    Computes and pushes adversarial validation dataframe.
    """
    def compute_adversarial_validation_df():
        features = feature_list
        n = 1
        dict_result = {}

        while True:
            adv_train = df_train[features].copy().astype('float32').fillna(0)
            adv_test = df_test[features].copy().astype('float32').fillna(0)
            adv_train['is_test'] = 0
            adv_test['is_test'] = 1

            adv_data = pd.concat([adv_train, adv_test])
            X_adv = adv_data[features]
            y_adv = adv_data['is_test']

            X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
                X_adv, y_adv, test_size=0.3, stratify=y_adv, random_state=random_state
            )

            clf = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_state
            )
            clf.fit(X_train_adv, y_train_adv)
            auc = roc_auc_score(y_test_adv, clf.predict_proba(X_test_adv)[:, 1])

            print(f"[Iter {n}] Adversarial validation AUC: {auc:.4f}")

            explainer = shap.TreeExplainer(clf)
            X_sample = X_adv.sample(n=100000, random_state=random_state)
            shap_values = explainer.shap_values(X_sample)[1]
            shap_importance = pd.DataFrame(shap_values, columns=X_sample.columns).abs().mean().sort_values(ascending=False)

            model_importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
            df_importance = pd.concat([model_importance, shap_importance], axis=1)
            df_importance.columns = ['model_importance', 'shap_importance']

            stable_features = shap_importance[shap_importance < shap_importance.quantile(threshold)].index.tolist()
            drop_features = [feature for feature in features if feature not in stable_features]

            iter_output = {
                'auc_score': auc,
                'feature_importance': df_importance.to_dict(orient='dict'),
                'drop_features': drop_features
            }

            dict_result[f'iter_{n}'] = iter_output

            if len(stable_features) == 0 or auc <= 0.5 or n >= max_iter:
                break

            features = stable_features
            n += 1

        return pd.DataFrame(dict_result).T

    return maybe_refresh_and_push(
        df_compute_fn=compute_adversarial_validation_df,
        filename="df_adversarial_validation.parquet",
        commit_msg="Add adversarial validation dataframe",
        github_token=github_token,
        refresh_repo_file=refresh_repo_file
    )