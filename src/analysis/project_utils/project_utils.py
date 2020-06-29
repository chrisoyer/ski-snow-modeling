#!/usr/bin/env python

"""Utilities for data cleaning and visualization
"""

import numpy as np
import pandas as pd
from itertools import product as it_product
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV

class SMWrapper(BaseEstimator, RegressorMixin):
    """wrap statsmodels ARIMA models to expose sklearn-style API"""

    def __init__(self, model_class, order, seasonal_order=None,
                 endog_col_name='base'):
        self.model_class = model_class
        self.order = order
        self.seasonal_order = seasonal_order
        self.endog_col_name = endog_col_name

    def fit(self, X=None, Y=None):
        self.endog = X[self.endog_col_name]
        self.exog_columns = [col for col in X.columns 
                             if col != self.endog_col_name]
        self.exog = X[self.exog_columns] if len(self.exog_columns) > 0 else None

        model_class_args = dict(endog=self.endog, exog=self.exog,
                                order=self.order,
                                seasonal_order=self.seasonal_order,)
        model_class_args = {k: v for k, v in model_class_args.items()
                            if v is not None}  # filter Nones even if series
        self.sm_model_ = self.model_class(**model_class_args)
        self.results_ = self.sm_model_.fit()
        self.resid = self.results_.resid
        self.plot_diagnostics = self.results_.plot_diagnostics
        self.mse, self.mae = self.results_.mse, self.results_.mae
        self.bic, self.aic = self.results_.bic, self.results_.aic

    def predict(self, X=None, steps=1, include_ci=False, **kwargs):
        self.X = X
        try:
            # only rows not included in original train dataset; don't include base
            self.endog_end = self.endog.shape[0]
            self.exog_regressors = self.X.iloc[self.endog_end:, 1:]
        except:
            print("should be sarima model")
            self.exog_regressors = None
        self.prediction_results = self.results_.get_prediction(
            start=self.X.iloc[0].name,
            end=self.X.iloc[-1].name,
            exog=self.exog_regressors,
            steps=steps,
            **kwargs)
        self.predictions = self.prediction_results.predicted_mean
        if include_ci:
            self.prediction_ci = self.prediction_results.conf_int()
        return self.predictions

    def summary(self):
        return self.results_.summary()

def resample_by_month(df, time_col):
    """resample df on monthly basis using specified timestamp column"""
    df = df.reset_index()
    # handling of time_col
    acceptable_time_col = ['timestamp', 'pseudo_ts']
    other_time_col = [x for x in acceptable_time_col if x != time_col].pop()
    if time_col not in acceptable_time_col:
        raise Exception(f'must be in {acceptable_time_col}')
    
    month = (df
             .set_index(f'{time_col}')
             .groupby([pd.Grouper(freq='1MS'), 'station'])
             .agg({'base': 'mean', 'snowfall': 'sum', 
                   'region': 'first', 'state': 'first', 'ski_yr': 'first',
                   f'{other_time_col}': 'first'})
            )
    return month


def index_setter(df=None, freq='M', index="pseudo_ts"):
    """sets index to just timestamp and sets frequency of timestamp"""
    try:
        df = df.reset_index(level='station')
    except KeyError:
        df = df.set_index(index)
    df.index = pd.DatetimeIndex(df.index.values, freq=freq)
    return df

def AC_PAC_plotter(df=None, lags=30, differences=None):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_acf(x=df.base,
                          lags=lags, alpha=0.05,
                          use_vlines=True, unbiased=False,
                          missing='conservative',
                          title=f'Autocorrelation: Differencing: {differences}',
                          zero=True, ax=ax1
                          )
    plot_pacf(x=df.base, lags=lags,
                           alpha=0.05, method='ldb', use_vlines=True,
                           title='Partial Autocorrelation', zero=True,
                           vlines_kwargs=None, ax=ax2)
    fig
    
def resid_plotter(residuals=None, y=None):
    """plot residual errors, exactly what it says on the tin"""
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
    residuals.plot(title="Residuals over Time", ax=ax1)
    residuals.plot(kind='kde', title="Residuals Distribution", ax=ax2)
    plt.show()
    print(residuals.describe())
    

def time_crossval(model, X, y):
    """takes sklearn-API model and returns CV results
    input:
        model: sklearn-API model
        X: 
        Y:
    """
    time_split = TimeSeriesSplit(8)
    cv_results = cross_validate(model, X, y, cv=time_split, 
                            scoring=['neg_root_mean_squared_error', 'r2',
                                     'neg_mean_absolute_error'], n_jobs=1) 
    return cv_results

def IC_chooser(unwrapped_mod=None, X=None, y=None, order_limit=(2,1)):
    """Use AIC/BIC to choose best model from grid of (pdq)(PDQ)S models
    Params:
        model:  sklearn model
        X:  
        y:
        order_limit:  tuple of highest order to use for pdq/PDQ
    Returns:
        df of AIC/BIC statistics
    """
    
    def pdq_maker(order_limit=2, s_order_limit=1, period=12):
        """generate all possible SARIMA orders for range"""
        p = d = q = range(0, order_limit)
        P = D = Q = range(0, s_order_limit)
        # Generate all different combinations of pdq/PDQ triplets
        pdq = list(it_product(p, d, q))
        seasonal_PDQ = [(x[0], x[1], x[2], period) for x in list(it_product(p, d, q))]
        return (pdq, seasonal_PDQ)

    pdq, seasonal_pdq = pdq_maker(*order_limit)
    AIC, BIC = {}, {}
    for order in pdq:
        for season_order in seasonal_pdq:
            model = SMWrapper(unwrapped_mod, order=order, seasonal_order=season_order)
            model.fit(X=X)
            AIC[f"{order}{season_order}12"] = model.aic
            BIC[f"{order}{season_order}12"] = model.bic
    ic_df = (pd.DataFrame.from_dict(AIC, orient='index').rename(columns={0:"AIC"})
             .merge(right=
                   pd.DataFrame.from_dict(BIC, orient='index').rename(columns={0:"BIC"}),
                   left_index=True, right_index=True)
             .sort_values(by='BIC', ascending=True)
             )
    return ic_df


def train_test_split(df=None, station=None, exog_cols=None, ski_yr_cutoff=7):
    """returns Train Test by year for one station
    Inputs:
        df: data source
        station: station to filter to
        exog_cols: list of enxogenous regressor column names
        ski_yr_cutoff: int of year to make split at
    returns:
        train_df, test_df
    """
    cols = ['base', 'ski_yr']
    if exog_cols:
        cols.extend(exog_cols)
    subset =  df.query('station==@station') \
    .pipe(index_setter, freq="MS",
          index='timestamp').fillna(0) \
    .filter(items=cols)
    train = subset.query('ski_yr<=@ski_yr_cutoff').drop(columns=['ski_yr'])
    test = subset.query('ski_yr>@ski_yr_cutoff').drop(columns=['ski_yr'])
    return train, test
    