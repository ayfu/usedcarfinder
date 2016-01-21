'''

__file__

    autotrader_compact.py

__description__

    This file provides utilities to grab data from MySQLdb and encode it
    choice between LabelEncoding and OneHotEncoder

'''

import sys
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import MySQLdb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split

sys.path.append(os.path.abspath("../web_scrape_sql/"))
from cglst_compact import *
sys.path.append(os.path.abspath("../web_scrape_sql/"))
from autotrader_compact import *

class dbConnect():
    '''
    Class to help with context management for 'with' statements.

    http://www.webmasterwords.com/python-with-statement

    c = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = 'nba_stats')
    with c:
        df.to_sql('nbadotcom', c.con, flavor = 'mysql', dtype = dtype)
    '''
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
    def __enter__(self):
        self.con = MySQLdb.connect(host = self.host, user = self.user,
                                   passwd = self.passwd, db = self.db)
        self.cur = self.con.cursor()
    def __exit__(self, type, value, traceback):
        self.cur.close()
        self.con.close()


class PruneLabelEncoder(LabelEncoder):
    '''
    Class variable that allows user to do traditional label encoding
    but they have the option to bin low frequency observations together
    '''
    def __init___(self):
        super(PruneLabelEncoder, self).__init__()
    def fit(self, series, cutoff=10):
        self.cutoff = cutoff
        # Generate the transformation classes and the map for low output munging
        super(PruneLabelEncoder, self).fit(series)
        trans_series = super(PruneLabelEncoder, self).transform(series)
        self.val_count_map = defaultdict(int)
        for i in trans_series:
            self.val_count_map[i] += 1
        # Identify the first key with low frequency and use it for low freq vals
        for key, val in self.val_count_map.items():
            if val < self.cutoff:
                self.low_cnt_target = key
                break
    def transform(self, series):
        trans_series = super(PruneLabelEncoder, self).transform(series)
        # Transform all the low frequency keys into the low frequency target key
        for key, val in self.val_count_map.items():
            if val < self.cutoff:
                trans_series[trans_series==key] = self.low_cnt_target
        return trans_series



class Encode():
    '''
    Initialize: which website (craigslist or autotrader) and db

    Returns: dataframe from MySQL that is encoded by label encoding,
             one hot encoding, or frequency encoding (use specified)
    '''

    def __init__(self, site = 'craigslist', db = 'find_car' ):
        self.site = site #initialize dataframe
        self.db = db

        # Make sure user input is craigslist or autotrader
        assert self.site == 'craigslist' \
            or self.site == 'autotrader', \
            'Misspelled craigslist or autotrader. User input: %s' % (site)

        # Connect with MySQL and grab data into one dataframe
        connect = dbConnect(host = 'localhost', user = 'root',
                            passwd = 'default', db = self.db)

        with connect:
            # Grab dataframes from SQL and concatenate them into one
            if self.site == 'craigslist':
                cg = ['chevrolet_cruze', 'ford_focus', 'honda_civic',
                      'hyundai_elantra', 'kia_forte', 'mazda_3',
                      'toyota_corolla']
                df_dict = {}
                for i in range(len(cg)):
                    cg[i] = cg[i]+'_cglst'
                    sql = 'SELECT * FROM %s;' % (cg[i])
                    df_dict[cg[i]] = pd.read_sql_query(sql,
                                                       con = connect.con,
                                                       index_col = 'index')

                self.df = pd.concat(df_dict.values(), axis = 0)
                self.df = self.df.reset_index().drop('index', axis = 1)

                # Filter bad titles and conditions
                self.df = self.df[self.df['title_stat'] != 'salvage']
                self.df = self.df[self.df['title_stat'] != 'missing']
                self.df = self.df[self.df['title_stat'] != 'lien']
                self.df = self.df[self.df['condition'] != 'salvage']

                # Filter ridiculous prices
                self.df = self.df[self.df['price'] > 1000]

            # Autotrader df
            else:
                at = ['chev', 'ford', 'honda', 'hyund', 'kia',
                      'mazda', 'toyota']
                df_dict = {}
                for i in range(len(at)):
                    at[i] = at[i]+'_autotrader'
                    sql = 'SELECT * FROM %s;' % (at[i])
                    df_dict[at[i]] = pd.read_sql_query(sql,
                                                       con = connect.con,
                                                       index_col = 'index')
                self.df = pd.concat(df_dict.values(), axis = 0)
                self.df = self.df.reset_index().drop('index', axis = 1)

    def label_encode(self, columns, TRANSFORM_CUTOFF = 0):
        '''
        Takes in: columns of interest and a cutoff value for bucketing
        encoding values

        Returns: Dataframe that is PruneEncoded for columns of interest
        '''

        # Check if there are 2 or more unique values in each columns
        for col in columns:
            if len(self.df[col].unique()) < 2:
                raise ValueError, 'Fewer than 2 unique values in %s' % (col)

        # Convert date column
        if self.site == 'craigslist':
            self.df['date'] = (self.df['date'] - self.df['date'].min())
            self.df['date'] = self.df['date'].astype(str)
            def date_convert(x):
                dates = x.split(' ')
                return dates[0]
            self.df['date'] = (self.df['date'].apply(date_convert)).astype(int)

        for col in columns:
            if type(self.df[col].unique()[1]) == str:
                le = PruneLabelEncoder()
                le.fit(self.df[col], TRANSFORM_CUTOFF)
                self.df[col] = le.transform(self.df[col])

        # Drop null values in year
        # Do linear regression prediction later for improved feature
        self.df = self.df[pd.notnull(self.df['year'])]

        # Place price column at the end
        price = self.df['price'].copy()
        self.df = self.df.drop(['price'], axis = 1)
        self.df['price'] = price
        print 'Finished Label Encoder'


    def onehot_encode(self, columns, TRANSFORM_CUTOFF = 0):
        '''
        Takes in: columns of interest

        Returns: Dataframe that one hot encodes the columns of interest
        '''

        # DEPRECATED pd.get_dummies(df[col]) is so much better
        '''
        self.label_encode(columns = self.df.columns,
                         TRANSFORM_CUTOFF = TRANSFORM_CUTOFF)
        '''

        # Convert date column
        if self.site == 'craigslist':
            self.df['date'] = (self.df['date'] - self.df['date'].min())
            self.df['date'] = self.df['date'].astype(str)
            def date_convert(x):
                dates = x.split(' ')
                return dates[0]
            self.df['date'] = (self.df['date'].apply(date_convert)).astype(int)

        # Drop null values in year
        # Do linear regression prediction later for improved feature
        self.df = self.df[pd.notnull(self.df['year'])]

        for col in columns:
            # DEPRECATED pd.get_dummies(df[col]) is so much better
            '''
            onehottemp = self.df[col].values
            lbl = OneHotEncoder()
            lbl.fit(np.resize(np.array(onehottemp),
                             (len(onehottemp), 1)))
            onehottemp = lbl.transform(np.resize(np.array(onehottemp),
                                      (len(onehottemp), 1))).toarray()
            for i in range(onehottemp.shape[1]):
                self.df[col + '_' + str(i)] = onehottemp[:,i]

            print lbl.get_params()
            '''
            onehottemp = pd.get_dummies(self.df[col])
            if sum(onehottemp.columns.isin(['unknown'])) > 0:
                self.df.rename(columns={'unknown':col+'_unknown'},
                               inplace=True)
            if sum(onehottemp.columns.isin(['other'])) > 0:
                self.df.rename(columns={'other':col+'_other'},
                               inplace=True)
            if sum(onehottemp.columns.isin(['hybrid'])) > 0:
                self.df.rename(columns={'hybrid':col+'_hybrid'},
                               inplace=True)
            if sum(onehottemp.columns.isin(['electric'])) > 0:
                self.df.rename(columns={'electric':col+'_electric'},
                               inplace=True)
            self.df = pd.concat([self.df,onehottemp], axis = 1)
            # Reformat column
            self.df = self.df.drop([col], axis = 1)

        # Place price column at the end
        price = self.df['price'].copy()
        self.df = self.df.drop(['price'], axis = 1)
        self.df['price'] = price
        print 'Finished OneHot Encoder'
