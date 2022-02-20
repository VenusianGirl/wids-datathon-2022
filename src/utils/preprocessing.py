import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def read_data(train_path, test_path):
    """Reads in train and test data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_duplicates(df, drop_cols=None):
    """Returns duplicated values in a dataframe."""
    if drop_cols is not None:
        return df[df.drop(columns=drop_cols).duplicated()]
    else:
        return df[df.duplicated()]


def remove_duplicates(df, drop_cols=None):
    """Removes duplicated values in a dataframe"""
    df_clean = df.copy()
    if drop_cols is not None:
        df_clean = df_clean[~df_clean.drop(columns=drop_cols).duplicated()]
    else:
        df_clean = df_clean[~df_clean.duplicated()]
    return df_clean.reset_index(drop=True)


def count_missing(df):
    """Counts the missing values in a dataframe"""
    missing_df = pd.DataFrame(
        df.isna().sum().sort_values(ascending=False), columns=["count"]
    )
    missing_df["percent"] = missing_df["count"] / df.shape[0]
    return missing_df.query("count != 0")


def iterative_impute(train_df, test_df, model, ct, target, feat_names, seed):
    """Iteratively impute missing values with a desired model"""
    train_imp = ct.fit_transform(train_df.drop(columns=[target]))
    test_imp = ct.transform(test_df)

    imputer = IterativeImputer(estimator=model, random_state=seed)

    # TODO: fix feat_names with appropriate sklearn method
    cols = (
        ct.named_transformers_["onehotencoder"].get_feature_names().tolist()
        + feat_names
    )

    train_imp = pd.DataFrame(imputer.fit_transform(train_imp), columns=cols)
    test_imp = pd.DataFrame(imputer.transform(test_imp), columns=cols)

    return train_imp, test_imp


def replace_columns(df, df_imp, columns):
    """Replace columns in one dataframe with columns from another"""
    df_replaced = df.copy()

    for col in columns:
        df_replaced[col] = df_imp[col]

    return df_replaced


def split_data(df, column, name):
    """Creates separate dataframes split on a single columns values"""
    dfs = {}

    for i in np.sort(df[column].unique()):
        split_df = df[df[column] == i]
        dfs[f"{name}_{i}"] = split_df

    return dfs


def create_X_y(dfs, target):
    """Separates train dfs into X and y datasets"""
    X_dfs = {}
    y_dfs = {}

    for name, df in dfs.items():
        X_dfs[name] = df.drop(columns=target)
        y_dfs[name] = df[target]

    return X_dfs, y_dfs
