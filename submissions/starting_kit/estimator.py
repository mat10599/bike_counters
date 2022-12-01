import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import datetime


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    X.loc[:, "weekend"] = X["weekday"] > 4

    X.loc[:, "day_of_the_year"] = X["date"].dt.day  # 2020 366
    X["sin_day_of_the_year"] = np.sin(2 * np.pi * X["day_of_the_year"] / 366)
    X["cos_day_of_the_year"] = np.cos(2 * np.pi * X["day_of_the_year"] / 366)

    X["sin_hours"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["cos_hours"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["sin_weekday"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["cos_weekday"] = np.cos(2 * np.pi * X["weekday"] / 7)

    X["sin_mnth"] = np.sin(2 * np.pi * X["month"] / 12)
    X["cos_mnth"] = np.cos(2 * np.pi * X["month"] / 12)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")

    categorical_cols = ["counter_name", "site_name", "weekday", "weekend"]

    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth"]
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
            # ("std_scaler", StandardScaler(), numerical_cols),
            ("passthrough", "passthrough", pass_through_cols),
        ],
    )

    # regressor = XGBRegressor(colsample_bytree=0.6, learning_rate=0.05, max_depth=7,
    #                         min_child_weight=4, min_split_loss=1, n_estimators=200, nthread=4, subsample=0.75)

    regressor = XGBRegressor()

    # xgb_grid = GridSearchCV(regressor,
    #                        parameters,
    #                        cv=2,
    #                        n_jobs=5,
    #                        verbose=True)

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe
