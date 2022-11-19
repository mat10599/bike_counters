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
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


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

    # rajouter # number of registered users if data is available?
    # The Autocorrelation Function (ACF) in R tells us the autocorrelation between current and lag values, and allows us to decide how many lag values to include in our model.
    # => check if adding count at t-1, -2, -3 is helpfull
    # wind, rain, temp, cloud

    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])

    X = pd.merge_asof(  # , "nbas" , "raf10"
        X.sort_values("date"), df_ext[["date", "hol_scol", "hol_bank", "t", "u", "rr1", "raf10", "nbas"]].sort_values("date").dropna(), on="date")  # , direction="nearest"
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    # train 0.481, valid 0.69, test 0.623, Bagged: valid 0.693, test 0.571
    categorical_cols = ["counter_name", "site_name",
                        "weekday", "weekend"]
    # train 0.435, valid 0.705, test 0.739, Bagged: valid 0.709, test 0.679 #WORSE
    # categorical_cols = ["counter_name", "site_name",
    #                    "weekday", "weekend", "hol_scol"]
    # train 0.467, valid 0.676, test 0.627, Bagged: valid 0.680, test 0.573 #bit worse
    categorical_cols = ["counter_name", "site_name",
                        "weekday", "weekend", "hol_bank"]

    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth"]

    # train 0.409, valid 0.705, test 0.652, Bagged: valid 0.708, test 0.585
    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth", "t"]
    # train 0.421, valid 0.686, test 0.617, Bagged: valid 0.690, test 0.569
    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth", "u"]
    # train 0.443, valid 0.676, test 0.625, Bagged: valid 0.681, test 0.574
    pass_through_cols = ["sin_hours", "cos_hours",
                         "sin_mnth", "cos_mnth", "rr1"]
    # train 0.432, valid 0.681, test 0.637, Bagged: valid 0.686, test 0.578
    pass_through_cols = ["sin_hours", "cos_hours",
                         "sin_mnth", "cos_mnth", "nbas"]
    # train 0.414, valid 0.692, test 0.646, Bagged: valid 0.696, test 0.584
    pass_through_cols = ["sin_hours", "cos_hours",
                         "sin_mnth", "cos_mnth", "raf10"]
    # train 0.39, valid 0.688, test 0.651, Bagged: valid 0.693, test 0.582
    pass_through_cols = ["sin_hours", "cos_hours",
                         "sin_mnth", "cos_mnth", "t", "u", "rr1", "nbas", "raf10"]

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

    # SCORE NO EXTERNAL DATA: train 0.481, valid 0.69, test 0.623, Bagged: valid 0.693, test 0.571
    # when merging data : train 0.481, valid 0.69, test 0.623, Bagged: valid 0.693, test 0.571 #not missing anything
    regressor = XGBRegressor()

    # BEST feat with this train 0.524, valid 0.71, test 0.604, Bagged: valid 0.713, test 0.567
    # With national holliday : train 0.515, valid 0.698, test 0.605, Bagged: valid 0.703, test 0.571
    # With nat hol and hum: train 0.483, valid 0.69, test 0.59, Bagged: valid 0.693, test 0.566
    # With nat hol t u: train 0.476, valid 0.718, test 0.601, Bagged: valid 0.722, test 0.572
    parameters = {  # 'nthread': [4],  # when use hyperthread, xgboost may become slower
        'learning_rate': [0.05],  # so called `eta` value
        'max_depth': [3, 5, 6, 7],
        'min_child_weight': [4],
        # 'min_split_loss': [1],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'n_estimators': [100, 200, 250]}

    xgb_grid = GridSearchCV(regressor,
                            parameters,
                            cv=2,
                            n_jobs=5,
                            verbose=True)

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
        # xgb_grid
    )

    return pipe
