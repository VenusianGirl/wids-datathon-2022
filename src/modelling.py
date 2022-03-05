import numpy as np
import pandas as pd
import lightgbm as lgb
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV


def get_ct_feat_names(ct, other_names):
    """Get the feature names from a ColumnTrasformer and combine
    with other feature names.

    Note: In recent versions of sklearn there is support for this
    without having to create a function.  However, Kaggle was running
    an older version of sklearn in their kernels during this competition,
    hence creation of this function.

    Parameters
    ----------
    ct : sklearn ColumnTransformer
        A fitted sklearn ColumnTransformer.
    other_names : list of str
        The other feature names to append

    Returns
    -------
    names : list of str
        The list of all feature names after a ColumnTransformer
        transforms a dataset.
    """
    names = []
    names += other_names
    return names


def logo_cv_lgb(
    model, X, y, groups, ct, early_stopping_rounds, return_models=True, other_names=None
):
    """Custom cross validation function for the WiDS 2022 Kaggle Competition.

    Parameters
    ----------
    model : lightGBM regressor
        The lgbm regressor model to perform cv with.
    X : pandas DataFrame
        X train set.
    y : pandas DataFrame
        y train set.
    groups : pandas DataFrame
        The leave one group out cv groups.
    ct : sklearn ColumnTransformer
        The column transformer for the dataset.
    early_stopping_rounds : int
        The number of early stopping rounds for lightGBM
    return_models : bool, optional
        Whether or not to return the indivdual lightGBM models, by default True
    other_names : list of str, optional
        Other feature names to append to CT names, by default None

    Returns
    -------
    results, models : pandas DataFrame, dict
        Results dataframe and dictionary of lgbm models.
    """

    models = []
    results = []
    train_scores = []
    val_scores = []
    fold = 0
    logo = LeaveOneGroupOut()

    for t, v in logo.split(X, groups=groups):
        fold += 1

        # to avoid lightgbm not saving new fit model (weird??? might be bug in lgb...)
        current_model = deepcopy(model)

        # train and val sets for current fold
        X_train, y_train = X.iloc[t], y.iloc[t]
        X_val, y_val = X.iloc[v], y.iloc[v]

        # year factors for current folds
        X_train_yf = np.sort(X_train["Year_Factor"].unique())
        X_val_yf = X_val["Year_Factor"].unique()

        # transform train and val sets
        X_train_tsf = ct.fit_transform(X_train)
        X_val_tsf = ct.transform(X_val)

        if other_names is None:
            feature_names = ct.get_feature_names()
        else:
            feature_names = get_ct_feat_names(ct, other_names)

        # fit and score model w/ early stopping
        current_model.fit(
            X_train_tsf,
            y_train,
            eval_set=[(X_val_tsf, y_val)],
            eval_metric=["rmse"],
            feature_name=feature_names,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        # train/val rmse scores
        train_rmse = mean_squared_error(
            y_true=y_train, y_pred=current_model.predict(X_train_tsf), squared=False
        )

        val_rmse = current_model.best_score_["valid_0"]["rmse"]

        fold_results = {
            "Fold": fold,
            "Train_shape": X_train_tsf.shape,
            "Val_shape": X_val_tsf.shape,
            "Train_years": X_train_yf,
            "Val_years": X_val_yf,
            "Train_RMSE": train_rmse,
            "Val_RMSE": val_rmse,
            "RMSE_spread": train_rmse - val_rmse,
        }

        results.append(fold_results)
        train_scores.append(train_rmse)
        val_scores.append(val_rmse)
        models.append(current_model)

    # mean validation score
    results.append(
        {
            "Fold": "MEAN",
            "Train_shape": "N/A",
            "Val_shape": "N/A",
            "Train_years": "N/A",
            "Val_years": "N/A",
            "Train_RMSE": np.mean(train_scores),
            "Val_RMSE": np.mean(val_scores),
            "RMSE_spread": np.mean(train_scores) - np.mean(val_scores),
        }
    )

    results = pd.DataFrame(results)

    if return_models:
        return results, models
    else:
        return results


def logo_cv_lgb_all(
    model,
    X_trains,
    y_trains,
    groups,
    ct,
    early_stopping_rounds,
    return_models=True,
    other_names=None,
):
    """Performs CV for each dataset in the WiDS kaggle competition."""
    results = {}
    models = {}

    for (n1, X), (n2, y), (n3, group) in zip(
        X_trains.items(), y_trains.items(), groups.items()
    ):
        assert n1 == n2 == n3
        results[n1], models[n1] = logo_cv_lgb(
            model, X, y, group, ct, early_stopping_rounds, return_models, other_names
        )

    return results, models


def process_cv_results(results):
    """Process cv results for multiple datasets into one dataframe"""
    results_df = pd.DataFrame()

    for name, result in results.items():
        result.insert(0, "facility_group", name)
        results_df = pd.concat([results_df, result])

    return results_df


def train_and_predict(model, X_trains, y_trains, X_tests, ct, target):
    """Train final WiDS models and get predictions.

    Parameters
    ----------
    model : sklearn estimator
        The model to train and predict.
    X_trains : dict
        Dictionary of X train sets.
    y_trains : dict
        Dictionary of y train sets.
    X_tests : dict
        Dictionary of X test sets.
    ct : sklearn ColumnTransformer
        Column transformer with tranformations to perform on data.
    target : str
        The target variable.

    Returns
    -------
    predictions : pandas DataFrame
        Dataframe containing the final predictions.
    """
    predictions = pd.DataFrame()

    for (n1, X_train), (n2, y_train), (n3, X_test) in zip(
        X_trains.items(), y_trains.items(), X_tests.items()
    ):
        assert n1 == n2 == n3

        current_model = deepcopy(model)
        pipe = make_pipeline(ct, current_model)

        pipe.fit(X_train, y_train)

        pred = {"id": X_test["id"], target: pipe.predict(X_test)}

        predictions = pd.concat([predictions, pd.DataFrame(pred)])

    predictions = predictions.sort_values("id")

    return predictions


def tune_hyperparameters(X_train, y_train, group, model, ct, param_dict, n_iter, score):
    """Tunes hyperparameters for a ML model with LOGO randomizedCV.

    Note: This function is not included in my final solution, as the time required
    to run it (even with GPU) takes much too long in Kaggle.

    Parameters
    ----------
    X_train : pandas DataFrame
        The X train dataset.
    y_train : pandas DataFrame
        The y train dataset.
    group : pandas DataFrame
        The groups for leave one group out CV.
    model : sklearn Estimator
        The model to tune
    ct : sklearn ColumnTransformer
        The column transformer to apply transformations to datasets.
    param_dict : dict
        The range for each hyperparameter for the search.
    n_iter : int
        The number of iterations of randomized search to perform.
    score : str
        The scoring metric.

    Returns
    -------
    sklearn RandomizedSearchCV
        Resulting object from the randomized search
    """
    logo = LeaveOneGroupOut()
    pipe = make_pipeline(ct, model)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dict,
        n_iter=n_iter,
        scoring=score,
        cv=logo,
    )

    search.fit(X_train, y_train, groups=group)

    return search


def tune_all_models(X_trains, y_trains, groups, models, ct, param_dicts, n_iter, score):
    """Performs hyperparameter tuning for each dataset and each model in WiDS 2022."""
    searches = {}

    for (n1, X_train), (n2, y_train), (n3, group) in zip(
        X_trains.items(), y_trains.items(), groups.items()
    ):
        assert n1 == n2 == n3
        for (n4, model), (n5, param_dict) in zip(models.items(), param_dicts.items()):
            assert n4 == n5
            search = tune_hyperparameters(
                X_train, y_train, group, model, ct, param_dict, n_iter, score
            )
            searches[f"{n1}_{n4}"] = search

    return searches
