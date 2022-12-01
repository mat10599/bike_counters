from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    X.loc[:, "weekend"] = X["weekday"] > 4

    X['sin_hours'] = np.sin(2*np.pi*X["hour"]/24)
    X['cos_hours'] = np.cos(2*np.pi*X["hour"]/24)

    X['sin_mnth'] = np.sin(2*np.pi*X["month"]/12)
    X['cos_mnth'] = np.cos(2*np.pi*X["month"]/12)

    return X.drop(columns=["date"])


def _merge_external_data(X):
    columns_in_merged = ["date", "cod_tend", "ff", "t", "u", "etat_sol", "hol_scol", "hol_bank", "quarantine1", "quarantine2",
                         "christmas_hols", "pmer", "tend", "td", "ww", "w1", "w2", "nbas", "raf10", "rafper",
                         "ht_neige"]
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    linear_cols = ["n", "nbas", "tend24", "raf10",
                   "ht_neige"]
    for col in linear_cols:
        df_ext[col] = df_ext[col].interpolate(
            method='linear', limit=4, limit_direction='both', axis=0)
    pad_cols = ["w1", "w2", "etat_sol"]
    for col in pad_cols:
        df_ext[col] = df_ext[col].interpolate(
            method='pad', limit=4, limit_direction='forward', axis=0)

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])

    X = pd.merge_asof(  # , "nbas" , "raf10"
        X.sort_values("date"), df_ext[columns_in_merged].sort_values("date").dropna(), on="date", direction="nearest")
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]

    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")

    categorical_cols = ["counter_name", "site_name", "year", "day", "hour", "month",
                        "weekend", "hol_scol", "hol_bank", "quarantine1",
                        "quarantine2", "christmas_hols"]

    pass_through_cols = ["weekday",
                         "sin_mnth", "cos_mnth", "sin_hours", "cos_hours"] + ["cod_tend", "ff", "t", "u", "etat_sol"]

    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
            ("std_scaler", StandardScaler(), pass_through_cols),
            #("passthrough", "passthrough", pass_through_cols)
        ],
    )

    regressor = CatBoostRegressor(logging_level="Silent")

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
