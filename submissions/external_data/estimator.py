from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from encode_date import _encode_date_both, _encode_date_with_transform_only, _encode_dates_without_transform



def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    X.loc[:, "day_of_the_year"] = X["date"].dt.day  # 2020 366
    X['sin_day_of_the_year'] = np.sin(2*np.pi*X["day_of_the_year"]/366)
    X['cos_day_of_the_year'] = np.cos(2*np.pi*X["day_of_the_year"]/366)

    X['sin_hours'] = np.sin(2*np.pi*X["hour"]/24)
    X['cos_hours'] = np.cos(2*np.pi*X["hour"]/24)
    # X.drop('hour', axis=1, inplace=True)

    # check si je peux enlever
    X['sin_weekday'] = np.sin(2*np.pi*X["weekday"]/7)
    X['cos_weekday'] = np.cos(2*np.pi*X["weekday"]/7)
    # X.drop('weekday', axis=1, inplace=True)

    X['sin_mnth'] = np.sin(2*np.pi*X["month"]/12)
    X['cos_mnth'] = np.cos(2*np.pi*X["month"]/12)
    # X.drop('month', axis=1, inplace=True)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])

    X = pd.merge_asof(
        # X.sort_values("date"), df_ext[["date", "t", "u", "vv", "nbas", "raf10", "rr1"]].sort_values("date").dropna(), on="date")
        X.sort_values("date"), df_ext[["date", "hol_scol", "hol_bank"]].sort_values("date").dropna(), on="date")
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name",
                        "year", "day", "hour", "weekday", "month", "hol_scol", "hol_bank"]

    numerical_cols = ["t", "u", "vv", "nbas", "raf10", "rr1"]

    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth",
                         "sin_weekday", "cos_weekday", "sin_day_of_the_year", "cos_day_of_the_year"]

    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
            # ("std_scaler", StandardScaler(), numerical_cols),
            ("passthrough", "passthrough", pass_through_cols)
        ],
    )

    regressor = Ridge()
    # regressor = MLPRegressor(hidden_layer_sizes=(
    #    8,), max_iter=200)  # max_iter = 1000
    #regressor = RandomForestRegressor(max_depth=15, n_estimators=15, n_jobs=-1)

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
