import numpy as np
import pandas as pd
import lightgbm as lgb
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut


def logo_cv_lgb(model, X, y, groups, ct, early_stopping_rounds, return_models=True):
    """Custom cross validation function meant for use with LightGBM Regressor"""

    models = []
    results = []
    train_scores = []
    val_scores = []
    fold = 0
    logo = LeaveOneGroupOut()

    for t, v in logo.split(X, groups=groups):
        fold += 1

        # to avoid lightgbm not saving new fit model (weird??? might be bug in lgb...)
        # might also be bug in my code?  to look into later
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

        # fit and score model w/ early stopping
        current_model.fit(
            X_train_tsf,
            y_train,
            eval_set=[(X_val_tsf, y_val)],
            eval_metric=["rmse"],
            feature_name=ct.get_feature_names(),
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
    model, X_trains, y_trains, groups, ct, early_stopping_rounds, return_models=True
):
    """Perform logo cv with LGBM for multiple datasets"""
    results = {}
    models = {}

    for (n1, X), (n2, y), (n3, group) in zip(
        X_trains.items(), y_trains.items(), groups.items()
    ):
        assert n1 == n2 == n3
        results[n1], models[n1] = logo_cv_lgb(
            model, X, y, group, ct, early_stopping_rounds, return_models
        )

    return results, models


def process_cv_results(results):
    """Process cv results for multiple datasets"""
    results_df = pd.DataFrame()

    for name, result in results.items():
        result.insert(0, "facility_group", name)
        results_df = pd.concat([results_df, result])

    return results_df
