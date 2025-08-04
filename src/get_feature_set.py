from src.github_push_file import push_parquet_to_github
from src.get_refresh_metadata import get_feature_drift_df
from src.get_refresh_metadata import get_correlation_train_df
from src.get_refresh_metadata import get_correlation_test_df
from src.get_refresh_metadata import get_autocorrelation_train_df
from src.get_refresh_metadata import get_adversarial_validation_df
from src.get_refresh_metadata import get_mutual_information_train_df
from src.get_refresh_metadata import get_vif_train_df

def get_feature_set(anonymized_features, corr_thresh = 'default', vif_tresh = 'default', mi_tresh = 'default'):
    """
    Returns selected feature set based on various threshold
    """
                   
    df_drift = get_feature_drift_df()
    df_corr_train = get_correlation_train_df()
    df_autocorr = get_autocorrelation_train_df()
    df_adv_val = get_adversarial_validation_df()
    df_mutual_information = get_mutual_information_train_df()
    df_vif = get_vif_train_df()

    if corr_thresh != 'default':
        corr_drop_features = set(df_corr_train[df_corr_train['corr'] >= corr_thresh].y)
        corr_filtered_features = [feature for feature in anonymized_features if feature not in corr_drop_features]
    else:
        corr_filtered_features = anonymized_features
    
    if vif_tresh != 'default':
        vif_filtered_features = df_vif[(df_vif['drop_vif'] < vif_tresh) | (df_vif['drop_vif'].isna())].feature.tolist()
    else:
        vif_filtered_features = anonymized_features
    
    if mi_tresh != 'default':
        mi_filtered_features = df_mutual_information[df_mutual_information['mutual_information'] > mi_tresh].feature.tolist()
    else:
        mi_filtered_features = anonymized_features

    selected_features = list(set(corr_filtered_features)
                             & set(vif_filtered_features)
                             & set(mi_filtered_features))

    print(f'Returning Features Set of Size: {len(selected_features)}')
    
    return selected_features