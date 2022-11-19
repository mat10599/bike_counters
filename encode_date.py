# file that takes the different functions to encode date
import numpy as np
import datetime
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries


def _encode_dates_without_transform(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def _encode_date_with_transform_only(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    X['sin_day_of_the_year'] = np.sin(2*np.pi*X["day"]/366)
    X['cos_day_of_the_year'] = np.cos(2*np.pi*X["day"]/366)

    X['sin_hours'] = np.sin(2*np.pi*X["hour"]/24)
    X['cos_hours'] = np.cos(2*np.pi*X["hour"]/24)
    # X.drop('hour', axis=1, inplace=True)

    # check si je peux enlever
    X['sin_weekday'] = np.sin(2*np.pi*X["weekday"]/7)
    X['cos_weekday'] = np.cos(2*np.pi*X["weekday"]/7)
    # X.drop('weekday', axis=1, inplace=True)

    X['sin_mnth'] = np.sin(2*np.pi*X["month"]/12)
    X['cos_mnth'] = np.cos(2*np.pi*X["month"]/12)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date", "year", "month", "day", "weekday", "hour"])

def _encode_date_both(X):

    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    X['sin_day_of_the_year'] = np.sin(2*np.pi*X["day"]/366)
    X['cos_day_of_the_year'] = np.cos(2*np.pi*X["day"]/366)

    X['sin_hours'] = np.sin(2*np.pi*X["hour"]/24)
    X['cos_hours'] = np.cos(2*np.pi*X["hour"]/24)
    # X.drop('hour', axis=1, inplace=True)

    # check si je peux enlever
    X['sin_weekday'] = np.sin(2*np.pi*X["weekday"]/7)
    X['cos_weekday'] = np.cos(2*np.pi*X["weekday"]/7)
    # X.drop('weekday', axis=1, inplace=True)

    X['sin_mnth'] = np.sin(2*np.pi*X["month"]/12)
    X['cos_mnth'] = np.cos(2*np.pi*X["month"]/12)
    
    
    return X.drop(columns=["date"])

