'''

__file__

    kmeans.py

__description__

    This file provides utilities to grab data from encode.py and do kMeans
    to generate new features

'''

import sys
import os
from collections import defaultdict
import datetime as dt

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from encode import *

# Parameters for kMeans clustering
params_km = {'n_clusters': 4,
             'max_iter': 1000,
             'n_init': 10,
             'init': 'k-means++',
             'precompute_distances': 'auto',
             'tol': 0.0001,
             'n_jobs': 1,
             'verbose': 0}

def km_cluster(df, params_km, n = 4, rounds = 100,
               columns = ['date','odometer','year','time']):
    '''
    Takes in dataframe (df), (n) number of clusters to explore,
    (rounds) number of rounds, and columns

    Returns a dataframe of the average and the standard deviation
    of value_counts()
    '''
    params_km['n_clusters'] = n
    dist = np.zeros(n)
    p = pd.DataFrame({'pred_0': dist})
    for x in range(rounds):
        est = KMeans(**params_km)
        X = df.as_matrix(columns).astype(float)
        est.fit(X)
        pred_km = est.predict(X)
        p['pred_'+str(x)] = np.array(pd.Series(pred_km).value_counts())
    results = pd.DataFrame({'mean': p.apply(np.mean, axis = 1),
                            'std': p.apply(np.std, axis = 1)})
    return results

def km_silhouette(df, params_km,
                  columns = ['date','odometer','year','time']):
    '''
    Takes in dataframe (df), (n) number of clusters to explore,
    (rounds) number of rounds, and columns

    Returns a dataframe of Silhouette Coefficients from kMeans
    '''

    cluster = np.arange(2,30)
    results = pd.DataFrame({'cluster': cluster})
    score = []
    for n in np.arange(2,30):
        params_km['n_clusters'] = n
        est = KMeans(**params_km)
        X = df.as_matrix(columns).astype(float)
        est.fit(X)
        labels = est.labels_
        score += [silhouette_score(X, labels, metric='euclidean')]
    results['score'] = score
    return results

def plot_silhouette(df, params_km, columns):
    """
    Takes in dataframe (df), (n) number of clusters to explore,
    (rounds) number of rounds, and columns

    Returns a plot of the Silhouette Coefficient vs. the
    number of clusters from kMeans Clustering
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize = (10, 8))

    km_df = km_silhouette(df, params_km, columns)

    ax.plot(km_df['cluster'],km_df['score'], color = 'black',
           label = 'Silhouette Coefficient',
           lw = 3, alpha = 0.6, ls = 'dashed')
    ax.set_xlabel('Number of kMeans Clusters',fontsize = 14)
    ax.set_ylabel('Silhouette Coefficient', fontsize = 14)
    ax.set_title('kMeans Analysis of Silhouette Coefficient', fontsize = 20)

def add_cluster(df, columns, params_km):
    '''
    Takes in: dataframe (df), columns to analyze, kMeans parameters (params_km)

    Returns: Dataframe with a new column for the cluster (to help categorize)
    '''
    est = KMeans(**params_km)
    X = df.as_matrix(columns).astype(float)
    est.fit(X)
    pred_km = est.predict(X)
    df['cluster'] = pred_km
    # put url and price at the end
    columns = list(df.columns)
    columns.remove('url')
    columns.append('url')
    columns.remove('price')
    columns.append('price')
    bigdf = bigdf[columns]
    return df
