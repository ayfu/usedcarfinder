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
import pickle

import pandas as pd
import numpy as np
import MySQLdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error

from encode import *
from kmeans import *


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
        split = round(self.df.shape[0]*self.test_size, 0)
        train = self.df.iloc[:self.df.shape[0]-split,:]
        test = self.df.iloc[self.df.shape[0]-split:,:]
        '''
        X = self.df.as_matrix(self.df.columns[:-1])
        y = self.df.as_matrix(['price'])[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=self.test_size,
                                                )
        '''
        # subset out URL column and price column
        self.train = train[train.columns[:-2]]
        self.test = test[test.columns[:-2]]
        self.X_train = train.as_matrix(self.df.columns[:-2])
        self.X_test = test.as_matrix(self.df.columns[:-2])
        self.y_train = train.as_matrix(['price'])[:,0]
        self.y_test = test.as_matrix(['price'])[:,0]
        self.url_train = train.as_matrix(['url'])[:,0]
        self.url_test = test.as_matrix(['url'])[:,0]

    def train_mdl(self, filename = 'model.pickle'):
        '''
        Takes in: file name

        Trains model on all data of interest, dumps the model, and
        saves it as a pickle file
        '''
        X = self.df.as_matrix(self.df.columns[:-2])
        y = self.df.as_matrix(['price'])[:,0]

        # Set up model
        if self.model == 'gb':
            self.mdl = GradientBoostingRegressor(**self.params)
        elif self.model == 'rf':
            self.mdl = RandomForestRegressor(**self.params)
        elif self.model == 'lasso':
            self.mdl = Lasso(**self.params)
        elif self.model == 'ridge':
            self.mdl = Ridge(**self.params)

        self.mdl.fit(X, y)

        # Pickle the model and dump it as a pickle file
        with open(filename,'wb') as f:
            pickle.dump(self.mdl, f)

    def kfold_cv(self, n_folds = 3):
        """
        Takes in: number of folds.

        Prints out RMSE score and stores the results in self.results
        """

        cv = KFold(n = self.X_train.shape[0], n_folds = n_folds)

        # Set up model
        if self.model == 'gb':
            self.mdl = GradientBoostingRegressor(**self.params)
        elif self.model == 'rf':
            self.mdl = RandomForestRegressor(**self.params)
        elif self.model == 'lasso':
            self.mdl = Lasso(**self.params)
        elif self.model == 'ridge':
            self.mdl = Ridge(**self.params)


        # Run through CV
        self.rmse_cv = []
        self.results = {'url': [],
                        'pred': [],
                        'real': []}

        for train, test in cv:
            self.mdl.fit(self.X_train[train], self.y_train[train])
            pred = self.mdl.predict(self.X_train[test])
            error = mean_squared_error(pred, self.y_train[test])**0.5
            self.results['pred'] += list(pred)
            self.results['real'] += list(self.y_train[test])
            self.results['url'] += list(self.url_train[test])
            self.rmse_cv += [error]
        print 'RMSE Scores:', self.rmse_cv
        print 'Mean RMSE:', np.mean(self.rmse_cv)

        # Store results of model
        # Work on avging results later
        if self.model == 'lasso' or self.model == 'ridge':
            coef = self.mdl.coef_
            self.coef_imp = pd.DataFrame({'feature': self.df.columns[:-2],
                                          'coefficient': coef})
            self.coef_imp = self.coef_imp.sort('coefficient', ascending = False)
            self.coef_imp = self.coef_imp.reset_index().drop('index', axis = 1)
            self.intercept = self.mdl.intercept_

        elif self.model == 'gb':
            feat = self.mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-2],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.estimators_ = self.mdl.estimators_
        else:
            feat = self.mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-2],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.oob_prediction_ = self.mdl.oob_prediction_
            self.estimators_ = self.mdl.estimators_


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
            self.mdl = GradientBoostingRegressor(**self.params)
        elif self.model == 'rf':
            self.mdl = RandomForestRegressor(**self.params)
        elif self.model == 'lasso':
            self.mdl = Lasso(**self.params)
        elif self.model == 'ridge':
            self.mdl = Ridge(**self.params)

        self.mdl.fit(self.X_train, self.y_train)
        self.preds = self.mdl.predict(self.X_test)
        self.rmse = mean_squared_error(self.preds, self.y_test)**0.5
        print 'RMSE score:', self.rmse
        #self.validate = pd.DataFrame(self.X_test, columns=self.df.columns[:-2])
        urls = pd.DataFrame(self.url_test, columns=['url'])
        pred = pd.DataFrame(self.preds, columns = ['pred'])
        price = pd.DataFrame(self.y_test, columns = ['price'])
        self.test = self.test.reset_index().drop('index', axis = 1)
        self.validate = pd.concat([self.test, urls, pred, price], axis = 1,
                                  join = 'inner')

        # Work on avging results later
        if self.model == 'lasso' or self.model == 'ridge':
            coef = self.self.mdl.coef_
            self.coef_imp = pd.DataFrame({'feature': self.df.columns[:-2],
                                          'coefficient': coef})
            self.coef_imp = self.coef_imp.sort('coefficient', ascending = False)
            self.coef_imp = self.coef_imp.reset_index().drop('index', axis = 1)
            self.intercept = self.self.mdl.intercept_

        elif self.model == 'gbr':
            feat = self.mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-2],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.estimators_ = self.mdl.estimators_
        else:
            feat = self.mdl.feature_importances_
            self.feat_imp = pd.DataFrame({'feature': self.df.columns[:-2],
                                          'importance': feat})
            self.feat_imp = self.feat_imp.sort('importance', ascending = False)
            self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
            self.oob_prediction_ = self.mdl.oob_prediction_
            self.estimators_ = self.mdl.estimators_

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
        ax.set_title('Results from Validation', fontsize = 30)
        ax.set_xlim(0,30000)
        ax.set_ylim(0,30000)
        ax.legend(loc=2, fontsize = 24)
        ax.tick_params(labelsize =20)

        ax.text(2500, 20000, 'GOOD DEAL REGION', fontsize=26)


def main():
    '''
    Train model and pickle dump it
    '''
    # Random Forest
    # Craigslist
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

    model_c = Model(df, model = 'rf', params = params_rf, test_size = 0)
    model_c.train_mdl(filename = 'cg_model.pickle')
    print 'Done with craigslist training'

    # Random Forest
    # Autotrader
    params_rf_a = {'n_estimators': 200,
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

    a = Encode(site = 'autotrader')
    a.label_encode(columns = a.df.columns, TRANSFORM_CUTOFF = 0)
    df_a = a.df

    model_a = Model(df_a, model = 'rf', params = params_rf_a, test_size = 0)
    model_a.train_mdl(filename = 'at_model.pickle')
    print 'Done with autotrader training'

    '''
    # VALIDATION TO SQL - both craigslist and autotrader

    # CRAIGSLIST FIRST
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
    columns = ['color','condition','drive','extra','fuel','model',
               'title_stat','transmission','type']
    c.label_encode(columns = columns, TRANSFORM_CUTOFF = 0)
    df = c.df

    val = Model(df, model = 'rf', params = params_rf, test_size = 0.3)
    val.validate()
    # Re-convert the categorical features back to text:
    for col in columns:
        val.validate[col] = c.le[col][0].inverse_transform(val.validate[col])

    # CREATE DEAL COLUMN
    """
    val.validate['deal'] = val.validate['pred'] - val.validate['price']
    connect = dbConnect(host = 'localhost', user = 'root',
              passwd = 'default', db = 'find_car')
    with connect:
            # Make sure dtypes fit for MySQL db
            dtype = {}
            for i in range(len(val.validate.columns)):
                if val.validate.columns[i] in ['condition', 'drive', 'extra',
                                               'fuel', 'model', 'color',
                                               'title_stat', 'transmission',
                                               'type','url']:
                    dtype[val.validate.columns[i]] = 'TEXT'
                elif val.validate.columns[i] in ['date','cylinders', 'year',
                                                 'time']:
                    dtype[val.validate.columns[i]] = 'INTEGER'
                else:
                    dtype[val.validate.columns[i]] = 'REAL'

            # Send to MySQL database, if table exists, continue to next
            val.validate.to_sql(name = 'result_cg', con = connect.con,
                      flavor = 'mysql', dtype = dtype)

    ############################################################################
    # AUTOTRADER
    ############################################################################
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

    a = Encode(site = 'autotrader')
    columns = ['model', 'condition', 'extra', 'type', 'color', 'fuel']
    a.label_encode(columns = columns, TRANSFORM_CUTOFF = 0)
    df = a.df

    val_a = Model(df, model = 'rf', params = params_rf, test_size = 0.3)
    val_a.validate()

    # Re-convert categorical features back to text:
    for col in columns:
        val_a.validate[col] = a.le[col][0].inverse_transform(val_a.validate[col])

    # CREATE DEAL COLUMN
    val_a.validate['deal'] = val_a.validate['pred'] - val_a.validate['price']


    connect = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = 'find_car')
    with connect:
            # Make sure dtypes fit for MySQL db
            dtype = {}
            for i in range(len(val_a.validate.columns)):
                if val_a.validate.columns[i] in ['color', 'condition', 'extra',
                                                 'model', 'type', 'fuel']:
                    dtype[val_a.validate.columns[i]] = 'TEXT'
                elif val_a.validate.columns[i] in ['url']:
                    dtype[val_a.validate.columns[i]] = 'MEDIUMTEXT'
                elif val_a.validate.columns[i] in ['odometer', 'year', 'citympg'
                                       'hwympg', 'cylinders']:
                    dtype[val_a.validate.columns[i]] = 'INTEGER'
                else:
                    dtype[val_a.validate.columns[i]] = 'REAL'

            # Send to MySQL database, if table exists, continue to next
            val_a.validate.to_sql(name = 'result_at', con = connect.con,
                      flavor = 'mysql', dtype = dtype)
        """
    '''


if __name__ == "__main__":
    main()
