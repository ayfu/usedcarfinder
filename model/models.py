'''

__file__

    models.py

__description__

    This file provides utilities to run models for Random Forest Regressor,
    Stochastic Gradient Boosting Regressor, Lasso Regression, and Ridge
    Regresion

'''


import sys
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import MySQLdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error

from encode import *


'''
Parameters for different Models
'''
# Random Forest
params_rf = {'n_estimators': 200,
                  'criterion': "mse",
                  'max_features': "auto",
                  'max_depth': None,
                  'min_samples_split': 2,
                  'min_samples_leaf': 2,
                  'min_weight_fraction_leaf': 0,
                  'oob_score': True,
                  #'max_leaf_notes': None
                  'verbose': 1
                  }
# Gradient Boosting
params_gb = {'loss': 'ls',
                   'learning_rate': 0.02,
                   'n_estimators': 100,
                   'max_depth': 5,
                   'min_samples_split': 5,
                   'min_samples_leaf': 3,
                   'subsample': 0.7
                  }
# Linear Regression
params_lin = {'fit_intercept': True,
              'normalize': False,
              'copy_X': True,
              'n_jobs': 1
              }
# Lasso Regression
params_lasso = {'alpha': 10,
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True}
# Ridge Regression
params_ridge = {'alpha': 0.4,
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True}


class Model():
    """
    Takes in: encoded dataframe, model to run, test_size

    Allows user to look at results of model

    Note: Please do feature engineering and filtering in the encode.py file
    """
    def __init__(self, df, model = 'gb', params = params_gb, test_size = 0.3):
        self.df = df
        self.test_size = 0.3
        self.params = params
        self.model = model
        assert self.model == 'gb' \
            or self.model == 'rf' \
            or self.model == 'lasso' \
            or self.model == 'ridge', \
            'Misspelled model (rf, gb, lasso, ridge).'+\
            'User input: %s' % (model)


        '''
        Performs a train_test_split after randomly mixing up the index
        of the dataset

        Sets attributes: X_train, X_test, y_train, y_test
        '''
        np.random.seed(1)
        self.df = self.df.reindex(np.random.permutation(self.df.index))
        self.df = self.df.reset_index().drop('index', axis = 1)
        X = self.df.as_matrix(self.df.columns[:-1])
        y = self.df.as_matrix(['price'])[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=self.test_size,
                                                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def kfold_cv(self, n_folds = 3):
        """
        Takes in: number of folds.

        Prints out RMSE score and stores the results in self.results
        """

        cv = KFold(n = self.X_train.shape[0], n_folds = n_folds)

        # Set up model
        if self.model == 'gb':
            mdl = GradientBoostingRegressor(**self.params)
        elif self.model == 'rf':
            mdl = RandomForestRegressor(**self.params)
        elif self.model == 'lasso':
            mdl = Lasso(**self.params)
        elif self.model == 'ridge':
            mdl = Ridge(**self.params)


        # Run through CV
        self.rmse_cv = []
        self.results = {'pred': [],
                        'real': []}

        for train, test in cv:
            mdl.fit(self.X_train[train], self.y_train[train])
            pred = mdl.predict(self.X_train[test])
            error = mean_squared_error(pred, self.y_train[test])**0.5
            self.results['pred'] += list(pred)
            self.results['real'] += list(self.y_train[test])
            self.rmse_cv += [error]
        print 'RMSE Scores:', self.rmse_cv
        print 'Mean RMSE:', np.mean(self.rmse_cv)

        # Store results of model
        # Work on avging results later
        if self.model == 'lasso' or self.model == 'ridge':
            coef = mdl.coef_
            self.coef_imp = pd.DataFrame({'feature': self.df.columns[:-1],
                                          'coefficient': coef})
            self.coef_imp = self.coef_imp.sort('coefficient', ascending = False)
            self.coef_imp = self.coef_imp.reset_index().drop('index', axis = 1)
            self.intercept = mdl.intercept_

        elif self.model == 'gb':
            feat = mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-1],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.estimators_ = mdl.estimators_
        else:
            feat = mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-1],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.oob_prediction_ = mdl.oob_prediction_
            self.estimators_ = mdl.estimators_


    def plot_results(self):
        """
        Plots results from CV
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (12,10))

        x1 = np.arange(0,30000)
        y1 = np.arange(0,30000)


        plt.fill_between(x1, y1, 30000, color=(0.01,0.40,0.1), alpha = 0.25)
        ax.scatter(self.results['real'], self.results['pred'],
                   color = (0.6,0.0,0.2), label = 'Model Predictions',
                   s = 100, alpha = 0.4)
        ax.plot(np.arange(0, 50000),np.arange(0, 50000), color = 'black',
                   label = 'Perfect Prediction Line',
                   lw = 4, alpha = 0.5, ls = 'dashed')

        ax.set_xlabel('Actual Price ($)',fontsize = 25)
        ax.set_ylabel('Predicted Price ($)', fontsize = 25)
        ax.set_title('Results from KFold Cross-Validation', fontsize = 30)
        ax.set_xlim(0,30000)
        ax.set_ylim(0,30000)
        ax.legend(loc=2, fontsize = 24)
        ax.tick_params(labelsize =20)

        ax.text(2500, 20000, 'GOOD DEAL REGION', fontsize=26)

    def validate(self):
        """
        Validate Model on Test set
        """
        # Set up model
        if self.model == 'gb':
            mdl = GradientBoostingRegressor(**self.params)
        elif self.model == 'rf':
            mdl = RandomForestRegressor(**self.params)
        elif self.model == 'lasso':
            mdl = Lasso(**self.params)
        elif self.model == 'ridge':
            mdl = Ridge(**self.params)

        mdl.fit(self.X_train, self.y_train)
        self.preds = mdl.predict(self.X_test)
        self.rmse = mean_squared_error(self.preds, self.y_test)**0.5
        print 'RMSE score:', self.rmse

        # Work on avging results later
        if self.model == 'lasso' or self.model == 'ridge':
            coef = self.mdl.coef_
            self.coef_imp = pd.DataFrame({'feature': self.df.columns[:-1],
                                          'coefficient': coef})
            self.coef_imp = self.coef_imp.sort('coefficient', ascending = False)
            self.coef_imp = self.coef_imp.reset_index().drop('index', axis = 1)
            self.intercept = self.mdl.intercept_

        elif self.model == 'gbr':
            feat = mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-1],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.estimators_ = mdl.estimators_
        else:
            feat = mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-1],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.oob_prediction_ = mdl.oob_prediction_
            self.estimators_ = mdl.estimators_

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (12,10))

        x1 = np.arange(0,30000)
        y1 = np.arange(0,30000)


        plt.fill_between(x1, y1, 30000, color=(0.01,0.40,0.1), alpha = 0.25)
        ax.scatter(self.y_test, self.preds, color = (0.6,0.0,0.2),
                   label = 'Model Predictions',
                   s = 100, alpha = 0.4)
        ax.plot(np.arange(0, 50000),np.arange(0, 50000), color = 'black',
                   label = 'Perfect Prediction Line',
                   lw = 4, alpha = 0.5, ls = 'dashed')

        ax.set_xlabel('Actual Price ($)',fontsize = 25)
        ax.set_ylabel('Predicted Price ($)', fontsize = 25)
        ax.set_title('Results from KFold Cross-Validation', fontsize = 30)
        ax.set_xlim(0,30000)
        ax.set_ylim(0,30000)
        ax.legend(loc=2, fontsize = 24)
        ax.tick_params(labelsize =20)

        ax.text(2500, 20000, 'GOOD DEAL REGION', fontsize=26)


def main():
    # Random Forest
    params_rf = {'n_estimators': 200,
                 'criterion': "mse",
                 'max_features': "auto",
                 'max_depth': None,
                 'min_samples_split': 2,
                 'min_samples_leaf': 2,
                 'min_weight_fraction_leaf': 0,
                 'oob_score': True,
                 #'max_leaf_notes': None
                 'verbose': 1
                  }

    c = Encode(site = 'craigslist')
    c.label_encode(columns = c.df.columns, TRANSFORM_CUTOFF = 0)
    df = c.df

    mdl = Model(df, model = 'rf', params = params_rf, test_size = 0.3)
    mdl.kfold_cv(n_folds = 6)

if __name__ == "__main__":
    main()
