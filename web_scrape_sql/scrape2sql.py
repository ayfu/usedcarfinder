'''

__file__

    scrape2sql.py

__description__

    This file uses the utilities in cglst_compact.py and atr_compact.py
    to scrape Craigslist and Autotrader, and then send the dataframes to
    a MySQL database

'''
import sys
import os
import datetime as dt
from collections import OrderedDict

import pandas as pd
import numpy as np
import MySQLdb

from cglst_compact import *
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


def main():
    '''
    Takes class variables from cglst_compact.py and populates MySQL db.


    '''
    all_cars = ['Honda+Civic', 'Toyota+Corolla', 'Ford+Focus',
                'Chevrolet+Cruze', 'Hyundai+Elantra', 'Mazda+3', 'Kia+Forte']
    all_cars2 = {'Honda': 'Civic', 'Toyota': 'Corol', 'Ford':'Focus',
                 'Chev': 'Cruze', 'Hyund': 'Elantr', 'Mazda':'Mazda3',
                 'Kia': 'Forte'}

    # Make connection to database
    connect = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = 'find_car')
    with connect:
        #'''
        # Craigslist
        for car in all_cars:
            print car
            # Format the name of the table
            tablename = car.lower().split('+')
            tablename = '_'.join(tablename)+'_cglst'

            # Run Scraper
            cglst = ScrapeCglst(model = car)
            df = cglst.all_data(cglst.url)

            # Make sure dtypes fit for MySQL db
            dtype = {}
            for i in range(len(df.columns)):
                if df.columns[i] in ['condition', 'drive', 'extra', 'fuel',
                                     'model', 'color', 'title_stat',
                                     'transmission', 'type','url']:
                    dtype[df.columns[i]] = 'TEXT'
                elif df.columns[i] == 'date':
                    dtype[df.columns[i]] = 'DATE'
                elif df.columns[i] in ['cylinders', 'year', 'time']:
                    dtype[df.columns[i]] = 'INTEGER'
                else:
                    dtype[df.columns[i]] = 'REAL'

            # Send to MySQL database, if table exists, continue to next
            try:
                df.to_sql(name = tablename, con = connect.con,
                      flavor = 'mysql', dtype = dtype)
                print
                print 'From www.craiglist.com'
                print 'Finished craigslist table for %s' % (car)
                print tablename
                print
            except:
                print '%s already exists' % (tablename)
        #'''
        #"""
        # Autotrader
        for car in all_cars2.keys():
            print car
            # Format the name of the table
            tablename = car.lower().split('+')
            tablename = '_'.join(tablename)+'_autotrader'

            # Run Scraper
            autotrader = ScrapeAutotrader(make = car, model = all_cars2[car])
            df = autotrader.all_data(autotrader.url)


            # Make sure dtypes fit for MySQL db
            dtype = {}
            for i in range(len(df.columns)):
                if df.columns[i] in ['color', 'condition', 'extra', 'model',
                                     'type', 'fuel']:
                    dtype[df.columns[i]] = 'TEXT'
                elif df.columns[i] in ['url']:
                    dtype[df.columns[i]] = 'MEDIUMTEXT'
                elif df.columns[i] in ['odometer', 'year', 'citympg'
                                       'hwympg', 'cylinders']:
                    dtype[df.columns[i]] = 'INTEGER'
                else:
                    dtype[df.columns[i]] = 'REAL'

            # Send to MySQL database, if table exists, continue to next
            try:
                df.to_sql(name = tablename, con = connect.con,
                      flavor = 'mysql', dtype = dtype)
                print
                print 'From www.autotrader.com'
                print 'Finished autotrader table for %s' % (car)
                print tablename
                print
            except:
                print '%s already exists' % (tablename)
        #"""
if __name__ == "__main__":
    main()
