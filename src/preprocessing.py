import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer


def read_data(train_path, test_path):
    """Read in train and test data for a kaggle competition.

    Parameters
    ----------
    train_path : str
        The path to the training data.
    test_path : str
        The path to the test data

    Returns
    -------
    train_df, test_df : pandas DataFrames
        The train and test datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_duplicates(df, drop_cols=None):
    """Determine and return the duplicated values in a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to check
    drop_cols : str or list of str, optional
        The columns to drop before returning duplicates.

    Returns
    -------
    pandas DataFrame
        A dataframe containing the rows with duplicated values.
    """
    if drop_cols is not None:
        return df[df.drop(columns=drop_cols).duplicated()]
    else:
        return df[df.duplicated()]


def remove_duplicates(df, drop_cols=None):
    """Removes the duplicated values in a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to check
    drop_cols : str or list of str, optional
        The columns to drop before removing duplicates.
    Returns
    -------
    pandas DataFrame
        A dataframe without duplicated.
    """
    df_clean = df.copy()

    if drop_cols is not None:
        df_clean = df_clean[~df_clean.drop(columns=drop_cols).duplicated()]
    else:
        df_clean = df_clean[~df_clean.duplicated()]

    return df_clean.reset_index(drop=True)


def count_missing(df):
    """Counts the missing data in a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to count the missing data in.

    Returns
    -------
    pandas DataFrame
        A summary of missing data (counts and %)
    """
    missing_df = pd.DataFrame(
        df.isna().sum().sort_values(ascending=False), columns=["count"]
    )
    missing_df["percent"] = missing_df["count"] / df.shape[0]
    return missing_df.query("count != 0")


def iterative_impute(train_df, test_df, model, ct, target, feat_names, seed):
    """Imputes missing data into train and test datasets with a ML model of choice.

    Parameters
    ----------
    train_df : pandas DataFrame
        The training dataset
    test_df : pandas DataFrame
        The test dataset
    model : sklearn estimator
        The machine learning model to use for imputation
    ct : sklearn ColumnTransformer
        The column transformer to perform on the dataset
    target : str
        The target variable (removed before imputation)
    feat_names : list
        Names of features to append to OHE features from column transformer.
    seed : int
        The random seed for imputation.

    Returns
    -------
    train_imp, test_imp : pandas DataFrames
        Train and test datasets with imputed values.
    """
    train_imp = ct.fit_transform(train_df.drop(columns=[target]))
    test_imp = ct.transform(test_df)

    imputer = IterativeImputer(estimator=model, random_state=seed)

    cols = (
        ct.named_transformers_["onehotencoder"].get_feature_names().tolist()
        + feat_names
    )

    train_imp = pd.DataFrame(imputer.fit_transform(train_imp), columns=cols)
    test_imp = pd.DataFrame(imputer.transform(test_imp), columns=cols)

    return train_imp, test_imp


def replace_columns(df, df_imp, columns):
    """Replace columns in a dataframe with columns from another.

    Note: Meant for use with imputed datasets for WiDS 2022.

    Parameters
    ----------
    df : pandas DataFrame
        The original dataframe.
    df_imp : pandas DataFrame
        The imputed dataframe.
    columns : str or list of str
        The columns to replace between dataframes.

    Returns
    -------
    df_replaced : pandas DataFrame
        Dataframe with replaced columns.

    """
    df_replaced = df.copy()

    for col in columns:
        df_replaced[col] = df_imp[col]

    return df_replaced


def impute_and_replace(
    train_dfs, test_dfs, model, ct, target, feat_names, replace, seed
):
    """Iteratively impute multiple dataframes.

    Note: Meant for use solely with WiDS 2022 data.  This simple calls
    the `iterative_impute` and `replace_column` functions for each
    individual dataset in my final WiDS solution.
    """
    train_dfs_imp = {}
    test_dfs_imp = {}

    # iterative imputation
    for (name1, train_df), (name2, test_df) in zip(train_dfs.items(), test_dfs.items()):
        train_imp, test_imp = iterative_impute(
            train_df, test_df, model, ct, target, feat_names, seed
        )
        train_dfs_imp[name1] = train_imp
        test_dfs_imp[name1] = test_imp

    # replace train columns with missing values
    for (name1, df), (name2, imp_df) in zip(train_dfs.items(), train_dfs_imp.items()):
        train_dfs[name1] = replace_columns(df, imp_df, replace)

    # replace test columns with missing values
    for (name1, df), (name2, imp_df) in zip(test_dfs.items(), test_dfs_imp.items()):
        test_dfs[name1] = replace_columns(df, imp_df, replace)

    return train_dfs, test_dfs


def split_data(df, column, name):
    """Splits a dataframe into multiple based on a categorical column.

    Note: This function was not used in my final WiDS solution.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe to split.
    column : str
        The column to split on.
    name : str
        The first part of the name for each dataframe.

    Returns
    -------
    dfs : dict of pandas DataFrame
        Dictionary of split dataframes.  Key: "name_#".  Value: the dataframe.
    """
    dfs = {}

    for i in np.sort(df[column].unique()):
        split_df = df[df[column] == i]
        dfs[f"{name}_{i}"] = split_df

    return dfs


def create_facility_groups(df):
    """Creates groups of facilities for the WiDS dataset based on their first word.

    Note: This function was not used in my final WiDS solution.

    Parameters
    ----------
    df : pandas DataFrame
        The WiDS train or test dataset.

    Returns
    -------
    dict of set
        The dictionary of facility types.
        The key is group name (first word in facility type).
        The value is a set containing the facility types in a given group.
    """
    groups_dict = defaultdict(list)
    groups_df = df.copy()
    groups_df["facility_first_word"] = groups_df["facility_type"].str.split(
        "_", expand=True
    )[0]

    for facility_type, g in (
        groups_df[["facility_type", "facility_first_word"]]
        .value_counts()
        .index.unique()
    ):
        groups_dict[g] += [facility_type]

    for g, l in groups_dict.items():
        groups_dict[g] = set(l)

    return dict(groups_dict)


def get_manual_facility_groups():
    """Returns the manual facility groups used in my final WiDS solution.

    Returns
    -------
    The dictionary of facility types.
        The key is group name (first word in facility type).
        The value is a set containing the facility types in a given group.
    """

    facility_groups = {
        "2to4_5plus_Mixed": {
            "2to4_Unit_Building",
            "5plus_Unit_Building",
            "Mixed_Use_Predominantly_Residential",
        },
        "Commercial_Education_Mixed_Industrial_Parking": {
            "Education_College_or_university",
            "Education_Other_classroom",
            "Education_Preschool_or_daycare",
            "Education_Uncategorized",
            "Commercial_Other",
            "Commercial_Unknown",
            "Mixed_Use_Commercial_and_Residential",
            "Mixed_Use_Predominantly_Commercial",
            "Industrial",
            "Parking_Garage",
        },
        "Food_Grocery": {
            "Food_Sales",
            "Food_Service_Other",
            "Food_Service_Restaurant_or_cafeteria",
            "Food_Service_Uncategorized",
            "Grocery_store_or_food_market",
        },
        "Health": {
            "Health_Care_Inpatient",
            "Health_Care_Outpatient_Clinic",
            "Health_Care_Outpatient_Uncategorized",
            "Health_Care_Uncategorized",
        },
        "Laboratory_Data": {"Laboratory", "Data_Center"},
        "Lodging": {
            "Lodging_Dormitory_or_fraternity_sorority",
            "Lodging_Hotel",
            "Lodging_Other",
            "Lodging_Uncategorized",
        },
        "Multifamily": {"Multifamily_Uncategorized"},
        "Office_Nursing": {
            "Office_Bank_or_other_financial",
            "Office_Medical_non_diagnostic",
            "Office_Mixed_use",
            "Office_Uncategorized",
            "Nursing_Home",
        },
        "Public": {
            "Public_Assembly_Drama_theater",
            "Public_Assembly_Entertainment_culture",
            "Public_Assembly_Library",
            "Public_Assembly_Movie_Theater",
            "Public_Assembly_Other",
            "Public_Assembly_Recreation",
            "Public_Assembly_Social_meeting",
            "Public_Assembly_Stadium",
            "Public_Assembly_Uncategorized",
            "Public_Safety_Courthouse",
            "Public_Safety_Fire_or_police_station",
            "Public_Safety_Penitentiary",
            "Public_Safety_Uncategorized",
        },
        "Religious": {"Religious_worship"},
        "Retail": {
            "Retail_Enclosed_mall",
            "Retail_Strip_shopping_mall",
            "Retail_Uncategorized",
            "Retail_Vehicle_dealership_showroom",
        },
        "Warehouse_Service": {
            "Warehouse_Distribution_or_Shipping_center",
            "Warehouse_Nonrefrigerated",
            "Warehouse_Refrigerated",
            "Warehouse_Selfstorage",
            "Warehouse_Uncategorized",
            "Service_Drycleaning_or_Laundry",
            "Service_Uncategorized",
            "Service_Vehicle_service_repair_shop",
        },
    }

    return facility_groups


def split_building_data(df, groups):
    """Splits the WiDS 2022 dataset based on groups of facility types.

    Parameters
    ----------
    df : pandas DataFrame
        The WiDS train or test dataset.
    groups : dict (key: str, value: set)
        The dictionary of facility types.
        The key should be the desired group name.
        The value should be a set containing the facility types in a given group.

    Returns
    -------
    dfs : dict
        A dictionary of pandas DataFrames, one for each group.
    """
    dfs = {}

    for name, group in groups.items():
        group_df = df.query("facility_type in @group")
        dfs[name] = group_df.reset_index(drop=True)

    return dfs


def create_X_y(dfs, target, group_col=None):
    """Separates each WiDS 2022 dataframe into X and y train sets.

    Parameters
    ----------
    dfs : dict
        A dictionary of pandas DataFrames, one for each group.
    target : str
        The target variable.
    group_col : str, optional
        The columns to use for CV groups, by default None

    Returns
    -------
    X_dfs, y_dfs, groups
        Dictionaries of X and y train sets, and groups.
    """

    X_dfs = {}
    y_dfs = {}
    groups = {}

    for name, df in dfs.items():
        X_dfs[name] = df.drop(columns=target)
        y_dfs[name] = df[target]

        if group_col is not None:
            groups[name] = df[group_col]

    if group_col is not None:
        return X_dfs, y_dfs, groups
    else:
        return X_dfs, y_dfs
