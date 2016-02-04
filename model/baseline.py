'''

__file__

    baseline.py

__description__

    This file provides utilities to calculate a baseline model. This baseline
    takes the average price for each model and each year with multiple entries

'''

import sys
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

from models import *

def df_cg():
    '''
    function that creates a new column 'avg' that takes the mean
    of each car for each year with multiple entries for craigslist
    '''
    cars = ['forte', 'civic', 'mazda3', 'cruze', 'corolla', 'elantra', 'focus']
    df_cg = Encode(site = 'craigslist')
    df_cg.df['avg'] = [0]*df_cg.df.shape[0]

    for car in cars:
        years = df_cg.df.loc[df_cg.df['model'] == car, 'year'].value_counts()
        years = years[years > 1].index
        for year in years:
            mean = np.mean(df_cg.df.loc[(df_cg.df['model'] == car) & \
                                        (df_cg.df['year'] == year),
                                        'price'])
            df_cg.df.loc[(df_cg.df['model'] == car) & \
                        (df_cg.df['year'] == year),
                        'avg'] = mean
    return df_cg.df

def df_at():
    '''
    function that creates a new column 'avg' that takes the mean
    of each car for each year with multiple entries for autotrader
    '''
    cars = ['forte', 'civic', 'mazda3', 'cruze', 'corolla', 'elantra', 'focus']
    df_at = Encode(site = 'autotrader')
    df_at.df['avg'] = [0]*df_at.df.shape[0]

    for car in cars:
        years = df_at.df.loc[df_at.df['model'] == car, 'year'].value_counts()
        years = years[years > 1].index
        for year in years:
            mean = np.mean(df_at.df.loc[(df_at.df['model'] == car) & \
                                        (df_at.df['year'] == year),
                                        'price'])
            df_at.df.loc[(df_at.df['model'] == car) & \
                        (df_at.df['year'] == year),
                        'avg'] = mean
    return df_at.df
