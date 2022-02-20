# TODO: Add final feature engineering functions here


def has_feature(df, feat):
    """Adds a boolean feature if a feature exists or now"""
    return df[feat].notna().astype(int)


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
