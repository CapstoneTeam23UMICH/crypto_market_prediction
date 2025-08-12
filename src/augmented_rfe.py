import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit


def augmented_rfe_timeseries_lightgbm(
    df,
    label_col='label',
    model_params=None,
    n_splits=5,
    num_boost_round=100,
    min_features=50,
    fold_weights=None,
    verbose=True
):
    if model_params is None:
        model_params = {
            "num_leaves":30,
            "max_depth": 5,
            "subsample": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "learning_rate": 0.05,
            "objective": "regression",
            "random_state": 42,
            "verbose": -1
        }

    X = df.drop(columns=[label_col])
    y = df[label_col].values
    features = list(X.columns)
    round_counter = 0
    feature_history = []
    protected_features = set()

    def run_cv(features):
        fold_scores = []
        fold_importances = []

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx][features]
            y_train = y[train_idx]
            X_val = X.iloc[val_idx][features]
            y_val = y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                model_params,
                train_set=train_data,
                valid_sets=[valid_data],
                num_boost_round=num_boost_round
            )

            preds = model.predict(X_val)
            score = pearsonr(y_val, preds)[0]
            fold_scores.append(score)
            fold_importances.append(model.feature_importance(importance_type='gain'))

            if verbose:
                print(f"Round {round_counter} | Fold {fold} | Val Pearson: {score:.4f}")

        return np.array(fold_scores), np.array(fold_importances)

    fold_scores, fold_importances = run_cv(features)

    while len(features) > min_features and len(protected_features) < len(features):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_counter} | Features: {len(features)}")
            print(f"{'='*60}")

        worst_folds = np.argsort(fold_scores)[:2]
        avg_importance = fold_importances[worst_folds].mean(axis=0)
        feature_to_importance = dict(zip(features, avg_importance))

        sorted_feats = sorted(feature_to_importance.items(), key=lambda x: x[1])
        for feat, _ in sorted_feats:
            if feat not in protected_features:
                candidate_feature = feat
                break
        else:
            if verbose:
                print("No more features to test.")
            break

        if verbose:
            print(f"Trying to drop: {candidate_feature}")

        test_features = [f for f in features if f != candidate_feature]
        new_fold_scores, _ = run_cv(test_features)

        weighted_improvement = sum(
            w for w, old, new in zip(fold_weights, fold_scores, new_fold_scores) if new > old
        )
        required_improvement = sum(fold_weights) // 2 

        if weighted_improvement >= required_improvement:
            if verbose:
                print(f"Dropped: {candidate_feature} (Weighted improvement = {weighted_improvement})")
            features = test_features
            fold_scores, fold_importances = new_fold_scores, _
        else:
            if verbose:
                print(f"Rejected: {candidate_feature} (Weighted improvement = {weighted_improvement})")
            protected_features.add(candidate_feature)

        weighted_mean_pearson = np.average(fold_scores, weights=fold_weights)

        feature_history.append({
            'round': round_counter,
            'features': features.copy(),
            'protected': protected_features.copy(),
            'weighted_mean_pearson': weighted_mean_pearson,
            'feature_importances': dict(zip(features, fold_importances.mean(axis=0))),
            'candidate': candidate_feature,
            'improvement_score': weighted_improvement
        })

        round_counter += 1

    return features, pd.DataFrame(feature_history)