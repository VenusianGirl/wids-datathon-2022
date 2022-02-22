# TODO: Add final feature engineering functions here


def has_feature(df, feat):
    """Adds a boolean feature if a feature exists or now"""
    return df[feat].notna().astype(int)


def feature_engineer(train_df, test_df):
    """Feature engineer the wids 2022 dataset"""
    train_df_eng = train_df.copy()
    test_df_eng = test_df.copy()

    # whether or not a building has a fog detector
    train_df_eng["has_fog_detector"] = has_feature(train_df_eng, "days_with_fog")
    test_df_eng["has_fog_detector"] = has_feature(test_df_eng, "days_with_fog")

    # whether or not a building has a wind detector
    train_df_eng["has_wind_detector"] = has_feature(train_df_eng, "max_wind_speed")
    test_df_eng["has_wind_detector"] = has_feature(test_df_eng, "max_wind_speed")

    return train_df_eng, test_df_eng


def feature_engineer_multiple(train_dfs, test_dfs):
    """Feature engineer multiple dataframes at once"""
    train_dfs_eng = {}
    test_dfs_eng = {}

    for (name1, train_df), (name2, test_df) in zip(train_dfs.items(), test_dfs.items()):
        assert name1 == name2
        train_dfs_eng[name1], test_dfs_eng[name2] = feature_engineer(train_df, test_df)

    return train_dfs_eng, test_dfs_eng


def add_cluster_labels(model, ct, train_df, test_df, target):
    """Adds cluster labels as a feature"""
    train_df_cluster = train_df.copy()
    test_df_cluster = test_df.copy()

    X_cluster = ct.fit_transform(train_df.drop(columns=[target]))
    test_cluster = ct.transform(test_df)

    model.fit(X_cluster)

    train_df_cluster["cluster_label"] = model.labels_
    test_df_cluster["cluster_label"] = model.predict(test_cluster)

    return train_df_cluster, test_df_cluster
