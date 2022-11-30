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
        X.sort_values("date"), df_ext[["date", "hol_bank", "hol_scol", "quarantine1", "quarantine2", "t", "rr1", "u", "nbas", "raf10"]].sort_values("date").dropna(), on="date")  # , direction="nearest"
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]

    # initialize list of lists
    data = [['100007049-102007049', 2.21044957472661],
            ['100056331-103056331', 6.246294046172539],
            ['100056331-104056331', 8.223815309842042],
            ['100056226-104056226', 17.757958687727825],
            ['100056327-104056327', 18.150425273390038],
            ['100056223-SC', 20.761846901579588],
            ['100056046-SC', 25.22430133657351],
            ['100047545-104047545', 26.383839611178615],
            ['100036719-104036719', 26.542041312272175],
            ['100060178-101060178', 27.980194410692587],
            ['100056226-103056226', 28.779465370595382],
            ['100056329-103056329', 30.693317132442285],
            ['100047545-103047545', 32.94228432563791],
            ['100056332-104056332', 33.22162818955042],
            ['100056329-104056329', 33.39599027946537],
            ['100036719-103036719', 34.23523693803159],
            ['100042374-110042374', 34.395369774919615],
            ['100049407-353255859', 34.86344844357976],
            ['100047546-103047546', 36.85577156743621],
            ['100056327-103056327', 37.868286755771564],
            ['100056332-103056332', 38.04811664641555],
            ['100049407-353255860', 38.32344357976654],
            ['100007049-101007049', 41.38517618469016],
            ['100047546-104047546', 42.229526123936814],
            ['100047547-103047547', 44.59380315917375],
            ['100056330-103056330', 45.349574726609966],
            ['300014702-353245972', 45.77198697068404],
            ['100057380-104057380', 46.03390448414145],
            ['100036718-104036718', 51.191737545565005],
            ['100047547-104047547', 52.23341433778858],
            ['100047548-103047548', 53.122114216281894],
            ['100047548-104047548', 54.68408262454435],
            ['100036718-103036718', 55.31445929526124],
            ['100063175-353277235', 58.53074119076549],
            ['100063175-353277233', 58.87108140947752],
            ['100042374-109042374', 65.87858520900322],
            ['100057329-104057329', 66.02187120291616],
            ['100056335-103056335', 67.72065613608748],
            ['100056336-106056336', 68.49246658566221],
            ['300014702-353245971', 68.68335333447625],
            ['100060178-102060178', 68.71749696233293],
            ['100056047-SC', 69.99076549210207],
            ['100056330-104056330', 75.74957472660996],
            ['100056334-104056334', 80.11300121506683],
            ['100056334-103056334', 80.77035236938032],
            ['100057329-103057329', 81.9805589307412],
            ['100047542-103047542', 92.24714459295261],
            ['100047542-104047542', 97.94301336573511],
            ['100057380-103057380', 97.95249088699879],
            ['100056335-104056335', 103.24714459295261],
            ['100056336-105056336', 107.98505467800729],
            ['100050876-103050876', 111.06792223572296],
            ['100044493-SC', 125.95078979343864],
            ['100057445-103057445', 164.99003645200486],
            ['100050876-104050876', 170.94775212636696],
            ['100057445-104057445', 219.8336573511543]]

    # Create the pandas DataFrame
    df_mean = pd.DataFrame(data, columns=['counter_id', 'mean'])

    # print(df_mean.head(30))
    # print(X.head())
    #X = X.merge(df_mean, on="counter_id")

    X["mean"] = X["counter_id"].replace(['100007049-102007049', '100056331-103056331',
                                         '100056331-104056331', '100056226-104056226',
                                         '100056327-104056327', '100056223-SC', '100056046-SC',
                                         '100047545-104047545', '100036719-104036719',
                                         '100060178-101060178', '100056226-103056226',
                                         '100056329-103056329', '100047545-103047545',
                                         '100056332-104056332', '100056329-104056329',
                                         '100036719-103036719', '100042374-110042374',
                                         '100049407-353255859', '100047546-103047546',
                                         '100056327-103056327', '100056332-103056332',
                                         '100049407-353255860', '100007049-101007049',
                                         '100047546-104047546', '100047547-103047547',
                                         '100056330-103056330', '300014702-353245972',
                                         '100057380-104057380', '100036718-104036718',
                                         '100047547-104047547', '100047548-103047548',
                                         '100047548-104047548', '100036718-103036718',
                                         '100063175-353277235', '100063175-353277233',
                                         '100042374-109042374', '100057329-104057329',
                                         '100056335-103056335', '100056336-106056336',
                                         '300014702-353245971', '100060178-102060178', '100056047-SC',
                                         '100056330-104056330', '100056334-104056334',
                                         '100056334-103056334', '100057329-103057329',
                                         '100047542-103047542', '100047542-104047542',
                                         '100057380-103057380', '100056335-104056335',
                                         '100056336-105056336', '100050876-103050876', '100044493-SC',
                                         '100057445-103057445', '100050876-104050876',
                                         '100057445-104057445'], [2.21044957472661, 6.246294046172539, 8.223815309842042,
                                                                  17.757958687727825, 18.150425273390038, 20.761846901579588,
                                                                  25.22430133657351, 26.383839611178615, 26.542041312272175,
                                                                  27.980194410692587, 28.779465370595382, 30.693317132442285,
                                                                  32.94228432563791, 33.22162818955042, 33.39599027946537,
                                                                  34.23523693803159, 34.395369774919615, 34.86344844357976,
                                                                  36.85577156743621, 37.868286755771564, 38.04811664641555,
                                                                  38.32344357976654, 41.38517618469016, 42.229526123936814,
                                                                  44.59380315917375, 45.349574726609966, 45.77198697068404,
                                                                  46.03390448414145, 51.191737545565005, 52.23341433778858,
                                                                  53.122114216281894, 54.68408262454435, 55.31445929526124,
                                                                  58.53074119076549, 58.87108140947752, 65.87858520900322,
                                                                  66.02187120291616, 67.72065613608748, 68.49246658566221,
                                                                  68.68335333447625, 68.71749696233293, 69.99076549210207,
                                                                  75.74957472660996, 80.11300121506683, 80.77035236938032,
                                                                  81.9805589307412, 92.24714459295261, 97.94301336573511,
                                                                  97.95249088699879, 103.24714459295261, 107.98505467800729,
                                                                  111.06792223572296, 125.95078979343864, 164.99003645200486,
                                                                  170.94775212636696, 219.8336573511543])
    # print(X.head())

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
    # train 0.409, valid 0.705, test 0.652, Bagged: valid 0.708, test 0.585
    categorical_cols = ["site_name",
                        "weekday", "weekend", "hol_bank",
                        "quarantine1", "quarantine2"]  # 0.63, 0.572
    # without site_name train  0.458  0.708 0.643 0.711 0.581
    # with site_name train  0.451  0.711 0.63 0.713 0.572

    # with counter_name : 0.451 0.711 0.63 0.713 0.572
    # with both : 0.447 0.698 0.619 0.701 0.580
    # with mean : 0.446 0.702 0.619 0.705 0.581

    # with counter_name : 0.451 0.711 0.63 0.713 0.572
    # with both : 0.447 0.698 0.619 0.701 0.580
    # with mean : 0.446 0.702 0.619 0.705 0.581

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

    # train 0.466, valid 0.675, test 0.63, Bagged: valid 0.679, test 0.574
    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth",
                         "cos_mnth"]
    # NEW weather but daily
    # train 0.383, valid 0.707, test 0.65, Bagged: valid 0.713, test 0.600
    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth",
                         "cos_mnth", "tavg", "tmin", "tmax", "prcp", "wspd"]
    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth",
                         "cos_mnth", "prcp", "wspd"]

    # train 0.394, valid 0.714, test 0.676, Bagged: valid 0.717, test 0.599
    #pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth","cos_mnth", "tmin"]
    # train 0.393, valid 0.708, test 0.675, Bagged: valid 0.711, test 0.612
    #pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth","cos_mnth", "tmax"]
    # train 0.41, valid 0.715, test 0.657, Bagged: valid 0.724, test 0.598
    #pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth","cos_mnth", "prcp"]
    # train 0.399, valid 0.71, test 0.654, Bagged: valid 0.714, test 0.596
    #pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth","cos_mnth", "wspd"]

    pass_through_cols = ["sin_hours", "cos_hours",
                         "sin_mnth", "cos_mnth", "t", "u", "rr1", "nbas", "raf10"]

    pass_through_cols = ["sin_hours", "cos_hours", "sin_mnth",
                         "cos_mnth", "mean"]

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
    # with all weather daily :
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
