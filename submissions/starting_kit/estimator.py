import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, PoissonRegressor, RidgeCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR

import datetime
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries


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

    #hol = SchoolHolidayDates()
    # X['holiday'] = X.apply(
    #    lambda row:  hol.is_holiday_for_zone(row["date"].date(), 'C') or
    #    JoursFeries.is_bank_holiday(row["date"].date(), zone="Métropole"), axis=1)

    # print(d.is_holiday_for_zone(X["date"].dt.date, 'C'))
    # X["holiday"] = d.is_holiday_for_zone(X["date"], 'C')

    # X['Holiday'] = X['Date'].isin(holidays)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name",
                        "year", "day", "hour", "weekday", "month"]  # "holiday"]

    # numerical_cols = ["latitude", "longitude"] #not helping

    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth",
                         "sin_day_of_the_year", "cos_day_of_the_year"]

    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
            # ("std_scaler", StandardScaler(), numerical_cols),
            ("passthrough", "passthrough", pass_through_cols)
        ],
    )

     #regressor = RidgeCV(alphas=[0.1, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15,
     #                  20, 25, 40, 60, 100, 200, 300, 500, 1000, 1500, 3000, 6000, 10000])
    #best_regressor_Ridge = RidgeCV(alphas = np.arange(0.1,6,0.5))   #best value is 4 when we dont include external data 
    # regressor = MLPRegressor(hidden_layer_sizes=(
    #    8,), max_iter=1000)  # max_iter = 1000
    #regressor = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    #regressor = PoissonRegressor()
    #regressor = SVR()
    #regressor = LinearSVR(max_iter=4000)
    #regressor = KNeighborsRegressor()
    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe

# All with Ridge()

# baseline
# valid  0.894
# test   0.723

# public holiday
# valid  0.886
# test   0.728

# full holiday
# valid
# test
