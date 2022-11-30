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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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

    X.loc[:, "weekend"] = X["weekday"] > 4

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

    # rajouter #week-end or working day #season
    # rajouter # number of registered users if data is available?
    # The Autocorrelation Function (ACF) in R tells us the autocorrelation between current and lag values, and allows us to decide how many lag values to include in our model.
    # => check if adding count at t-1, -2, -3 is helpfull
    # wind, rain, temp, cloud

    # hol = SchoolHolidayDates()
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

    # train 0.401 valid  0.75 test 0.685 Bagged scores 0.754 0.603
    categorical_cols = ["counter_name", "site_name",
                        "year", "day", "hour", "weekday", "month"]  # "holiday"]

    # train 0.62 valid  0.768 test 0.69 Bagged scores 0.772 0.643
    categorical_cols = ["counter_name", "site_name"]
    # train 0.62 valid  0.772 test 0.718 Bagged scores 0.776 0.657
    categorical_cols = ["counter_name", "site_name", "year"]
    # train 0.46 valid  0.847 test 0.825 Bagged scores 0.852 0.745
    categorical_cols = ["counter_name", "site_name", "day"]
    # train 0.621 valid  0.771 test 0.705 Bagged scores 0.776 0.641
    categorical_cols = ["counter_name", "site_name", "hour"]
    # train 0.484 valid  0.695 test 0.626 Bagged scores 0.698 0.572
    categorical_cols = ["counter_name", "site_name", "weekday"]
    # train 0.481 valid  0.69 test 0.623 Bagged scores 0.693 0.571
    categorical_cols = ["counter_name",
                        "weekday", "weekend"]  # 0.63, 0.713, 0.572
    # train 0.481 valid  0.692 test 0.639 Bagged scores 0.695 0.577
    # categorical_cols = ["counter_name",
    #                    "site_name", "weekday", "weekend", "year"]  # CHECK PQ YEAR n'a pas d'influence positive

    # numerical_cols = ["latitude", "longitude"] #doesn't help

    # train 0.453 valid  0.801 test 0.71 Bagged scores 0.810 0.659
    pass_through_cols = []
    # train 0.406 valid  0.788 test 0.722 Bagged scores 0.800 0.671
    pass_through_cols = ["sin_hours", "cos_hours"]
    # train 0.401 valid  0.75 test 0.685 Bagged scores 0.754 0.603
    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth"]
    # train 0.385, valid 0.741, test 0.715, Bagged: valid 0.746, test 0.638
    # pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth", "cos_mnth",
    #                     "sin_day_of_the_year", "cos_day_of_the_year"]

    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
            # ("std_scaler", StandardScaler(), numerical_cols),
            ("passthrough", "passthrough", pass_through_cols)
        ],
    )

    # regressor = RidgeCV(alphas=[0.1, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15,
    #                  20, 25, 40, 60, 100, 200, 300, 500, 1000, 1500, 3000, 6000, 10000])
    # best_regressor_Ridge = RidgeCV(alphas = np.arange(0.1,6,0.5))   #best value is 4 when we dont include external data
    # regressor = MLPRegressor(hidden_layer_sizes=(
    #    8,), max_iter=1000)  # max_iter = 1000
    # regressor = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    # regressor = PoissonRegressor()
    # regressor = SVR()
    # regressor = LinearSVR(max_iter=4000)
    # regressor = KNeighborsRegressor()

    regressor = LGBMRegressor()

    parameters = {  # 'nthread': [4],  # when use hyperthread, xgboost may become slower
        'learning_rate': [.03, 0.05, .07],  # so called `eta` value
        'max_depth': [3, 5, 6, 7],
        'min_child_weight': [4],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'n_estimators': [100, 200, 300]}  # THIS IS THE ONE I GOT THE BEST SCORE WITH

    # BEST feat with this train 0.524, valid 0.71, test 0.604, Bagged: valid 0.713, test 0.567
    parameters = {  # 'nthread': [4],  # when use hyperthread, xgboost may become slower
        'learning_rate': [0.05],  # so called `eta` value
        'max_depth': [3, 5, 6, 7],
        'min_child_weight': [4],
        # 'min_split_loss': [1],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'n_estimators': [100, 200, 250]}

    # train  0.192 valid 0.745 test 0.735
    regressor = XGBRegressor(max_depth=10, n_estimators=300)
    # train  0.371 valid  0.739 test   0.698
    regressor = XGBRegressor(max_depth=5, n_estimators=200)
    # train  0.497 valid  0.767 test  0.681
    regressor = XGBRegressor(max_depth=3, n_estimators=200)
    # train  0.335 valid  0.743 test 0.7
    regressor = XGBRegressor(max_depth=5, n_estimators=400)
    # train  0.393 valid  0.738 test 0.696
    regressor = XGBRegressor(max_depth=5, n_estimators=150)
    # train  0.428 valid  0.74 test 0.695 BEST
    regressor = XGBRegressor(max_depth=5, n_estimators=100)
    # train  0.491 valid  0.754 test 0.7
    # regressor = XGBRegressor(max_depth=5, n_estimators=50)
    # train  0.448 valid  0.745 test 0.697
    # regressor = XGBRegressor(max_depth=5, n_estimators=80)

    regressor = XGBRegressor()
    xgb_grid = GridSearchCV(regressor,
                            parameters,
                            cv=2,
                            n_jobs=5,
                            verbose=True)

    # train  0.467 valid  0.739  test   0.664 , Bagged : valid 0.745, test 0.622
    regressor = XGBRegressor(colsample_bytree=0.6, learning_rate=0.05, max_depth=7,
                             min_child_weight=4, min_split_loss=1, n_estimators=200, nthread=4, subsample=0.75)

    # train 0.385, valid 0.741, test 0.715, Bagged: valid 0.746, test 0.638
    regressor = XGBRegressor()

    #pipe = make_pipeline(date_encoder, preprocessor, regressor)

    pipe = make_pipeline(date_encoder, preprocessor, xgb_grid)

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
