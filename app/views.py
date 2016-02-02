import os
from contextlib import closing
from collections import OrderedDict
import re
from collections import defaultdict
import pickle
import datetime

from flask import Flask, request, session, g, redirect, url_for, \
                  abort, render_template, flash, make_response
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import MySQLdb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error

from app import app

# SET TRANSFORM_CUTOFF to 0
TRANSFORM_CUTOFF = 0


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


"""
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
"""
@app.route('/', methods = ['GET', 'POST'])
@app.route('/home', methods = ['GET', 'POST'])
def index():
    user = '' # fake user
    url = "http://sfbay.craigslist.org/pen/ctd/5413262011.html"
    return render_template("index.html",
                            title = 'Home',
                            user = user,
                            url = url)





@app.route('/carcheck', methods = ['GET', 'POST'])
def carcheck():

    #ALL = request.args.get('ALL')
    # Monitor if url is good
    bad_url = 0

    ####
    # FIX THIS NONSENSE WITH SITE = CRAIGSLIST

    ####


    if request.method == 'POST':
        urlcall = request.form['urlcall']
        url = urlcall
        if re.search('craigslist', url):
            site = 'craigslist'
            data = {'model': [],
                    'price': [],
                    'year': [],
                    'extra': [],
                    'fuel': [],
                    'cylinders': [],
                    'odometer': [],
                    'transmission': [],
                    'title_stat': [],
                    'drive': [],
                    'type': [],
                    'color': [],
                    'condition': [],
                    'time':[],
                    'date':[]
                   }
            try:
                html = requests.get(url)
                if html == None:
                    reason = 'Bad URL'
                    return render_template("invalid.html", reason = reason)
            except:
                reason = 'Bad URL'
                return render_template("invalid.html", reason = reason)
        elif re.search('autotrader', url):
            site = 'autotrader'
            hdr = {'Host': 'www.autotrader.com',
                   'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; ' +\
                                 'rv:43.0) Gecko/20100101 Firefox/43.0'
                   }
            data = {'model': [],
                    'condition': [],
                    'year': [],
                    'extra': [],
                    'type': [],
                    'odometer': [],
                    'color': [],
                    'hwympg': [],
                    'citympg': [],
                    'price': [],
                    'fuel': [],
                    'cylinders': []
                    #'description': []
                   }
            try:
                html = requests.get(url, headers = hdr)
            except:
                reason = 'Bad URL'
                return render_template("invalid.html", reason = reason)
        else:
            reason = 'Bad URL'
            return render_template("invalid.html", reason = reason)

        if site == 'craigslist':
            soup = BeautifulSoup(html.text, 'lxml')
            dat = soup.findAll('p', attrs={'class':"attrgroup"})
            if len(dat) < 1:
                reason = 'Bad URL'
                return render_template("invalid.html", reason = reason)

            # Take care of price, time, and date first
            # try statement for one anomolous page for Mazda 3
            time = soup.find('time',
                             attrs={'datetime': re.compile('.*')}).get_text()

            def time_convert(x):
                atimes = x.split(':')
                return (3600*int(atimes[0])+60*int(atimes[1]))+int(atimes[2])
            time = time.split(' ')
            data['date'] += [time[0]]
            # Convert time string into datetime and then back to formatted time
            time_temp = pd.DataFrame({'time':[time[1]]})
            time_temp = pd.to_datetime(time_temp['time'])
            time[1] = time_temp.dt.time.astype(str)[0]
            data['time'] += [time_convert(time[1])]




            price = soup.find('span', attrs={'class': 'price'})
            price_check = 0 # if price_check = 1, means null value
            if price == None:
                # FILTER OUT LATER BECAUSE OF RIDICULOUSNESS
                data['price'] = 0
                price_check = 1
            else:
                price = price.get_text()
                data['price'].append(float(price.split('$')[-1]))


            # Checks if title contains car type (ie. hatchback)
            hatch = 0
            sedan = 0
            coupe = 0


            title = soup.title.get_text()
            print title

            # Determine car model
            query = re.search('[cC][iI][vV][iI][cC]|' +\
                              '[cC][oO][rR][oO][lL][lL][aA]|' +\
                              '[fF][oO][cC][uU][sS]|' +\
                              '[cC][rR][uU][zZ][eE]|' +\
                              '[eE][lL][aA][nN][tT][rR][aA]|' +\
                              '[fF][oO][rR][tT][eE]|' +\
                              '[mM][aA][zZ][dD][aA]', title)

            # Making sure we get SOMETHING
            if query == None:
                reason = 'Did not detect civic, corolla, focus, cruze' +\
                         ', elantra, forte, or mazda'
                return render_template("invalid.html", reason = reason)
            query = query.group(0).lower()

            # Fill in first features (from webpage or from title)
            if dat[0].span.get_text() != None:
                # year is from title
                year = dat[0].span.get_text()
                year1 = re.search('([^a-zA-Z0-9:\.*_]\s*[12][09][0-9][0-9]'+\
                                 '[^a-zA-Z0-9~_\-\.]|'+\
                                 '^[12][09][0-9][0-9][^a-zA-Z0-9~_\-\.])',
                                 year)
                year2 = re.search('([^a-zA-Z0-9:\.*_]\s*[12][09][0-9][0-9]'+\
                                 '[^a-zA-Z0-9~_\-\.]|'+\
                                 '^[12][09][0-9][0-9][^a-zA-Z0-9~_\-\.])',
                                 title)
                if year1:
                    data['year'] += [int(year1.group(0))]
                elif year2:
                    data['year'] += [int(year2.group(0))]
                else:
                    reason = 'No reported year'
                    return render_template("invalid.html", reason = reason)

                # Different Cars - scrape 'extra' feature from title

                # CIVIC
                if query == 'civic':
                    data['model'] += ['civic']

                    hybrid = re.search('([hH][yY][bB][rR][iI][dD])', title)
                    ex = re.search('([eE][xX])', title)
                    lx = re.search('([lL][xX])', title)
                    si = re.search('([sS][iI])', title)
                    if hybrid:
                        data['extra'] += ['hybrid']
                    elif ex:
                        data['extra'] += ['ex']
                    elif lx:
                        data['extra'] += ['lx']
                    elif si:
                        data['extra'] += ['si']
                    else:
                        data['extra'] += [np.nan]

                # COROLLA
                if query == 'corolla':
                    data['model'] += ['corolla']

                    l = re.search('\s+[lL]\s*', title)
                    le = re.search('\s+[lL][eE]\s*', title)
                    leco = re.search('[lL][eE]\s+[eE][cC][oO]', title)
                    s = re.search('\s+[sS]\s*', title)
                    if le:
                        data['extra'] += ['le']
                    elif leco:
                        data['extra'] += ['leEco']
                    elif s:
                        data['extra'] += ['s']
                    elif l:
                        data['extra'] += ['l']
                    else:
                        data['extra'] += [np.nan]

                # CRUZE
                if query == 'cruze':
                    data['model'] += ['cruze']
                    l = re.search('\s+[lL]\s*', title)
                    ls = re.search('\s+[lL][sS]\s*', title)
                    lt1 = re.search('\s+1[lL][tT]\s*', title)
                    lt2 = re.search('\s+2[lL][tT]\s*', title)
                    ltz = re.search('\s+[lL][tT][zZ]\s*', title)
                    eco = re.search('\s+[eE][cC][oO]\s*', title)
                    if eco:
                        data['extra'] += ['eco']
                    elif ltz:
                        data['extra'] += ['ltz']
                    elif lt2:
                        data['extra'] += ['2lt']
                    elif lt1:
                        data['extra'] += ['1lt']
                    elif ls:
                        data['extra'] += ['ls']
                    elif l:
                        data['extra'] += ['l']
                    else:
                        data['extra'] += [np.nan]

                # ELANTRA
                if query == 'elantra':
                    data['model'] += ['elantra']
                    gt = re.search('\s+[gG][tT]\s*', title)
                    gls = re.search('\s+[gG][lL][sS]\s*', title)
                    se = re.search('\s+[sS][eE]\s*', title)
                    sport = re.search('\s+[sS][pP][oO][rR][tT]\s*',
                                      title)
                    lmt = re.search('\s+[lL][iI][mM][iI][tT][eE][dD]\s*',
                                    title)
                    if lmt:
                        data['extra'] += ['limited']
                    elif sport:
                        data['extra'] += ['sport']
                    elif se:
                        data['extra'] += ['se']
                    elif gls:
                        data['extra'] += ['gls']
                    elif gt:
                        data['extra'] += ['gt']
                    else:
                        data['extra'] += [np.nan]

                # MAZDA 3
                if query == 'mazda':
                    data['model'] += ['mazda3']
                    data['extra'] += [np.nan]

                # KIA FORTE
                if query == 'forte':
                    data['model'] += ['forte']
                    lx = re.search('\s+[lL][xX]\s*', title)
                    ex = re.search('\s+[eE][xX]\s*', title)
                    sx = re.search('\s+[sS][xX]\s*', title)
                    if lx:
                        data['extra'] += ['lx']
                    elif ex:
                        data['extra'] += ['ex']
                    elif sx:
                        data['extra'] += ['sx']
                    else:
                        data['extra'] += [np.nan]

                # FOCUS
                if query == 'focus':
                    data['model'] += ['focus']
                    s = re.search('\s+[sS]\s*', title)
                    s_sed = re.search('\s+[sS]\s+[sS][eE][dD][aA][nN]',
                                      title)
                    s_hat = re.search('\s+[sS]\s+[hH][aA][tT][cC][hH]',
                                      title)
                    se = re.search('\s+[sS][eE]\s*', title)
                    ses = re.search('\s+[sS][eE][sS]\s*', title)
                    sel = re.search('\s+[sS][eE][lL]\s*', title)
                    svt = re.search('\s+[sS][vV][tT]\s*', title)
                    se_sed = re.search('\s+[sS][eE]\s+[sS][eE][dD][aA][nN]',
                                       title)
                    se_hat = re.search('\s+[sS][eE]\s+[hH][aA][tT][cC][hH]',
                                       title)
                    ti = re.search('\s+[tT][iI][tT][aA][nN][iI][uU][mM]\s*',
                                   title)
                    ti_sed = re.search('\s+[tT][iI][tT][aA][nN][iI][uU][mM]'+\
                                       '\s+[sS][eE][dD][aA][nN]', title)
                    ti_hat = re.search('\s+[tT][iI][tT][aA][nN][iI][uU][mM]'+\
                                       '\s+[hH][aA][tT][cC][hH]', title)
                    st = re.search('\s+[sS][tT]\s*', title)
                    el = re.search('\s+[eE][lL][eE][cC][tT][rR][iI][cC]\s*',
                                   title)
                    rs = re.search('\s+[rR][sS]\s*', title)
                    if rs:
                        data['extra'] += ['rs']
                    elif el:
                        data['extra'] += ['electric']
                    elif st:
                        data['extra'] += ['st']
                    elif ti_hat:
                        data['extra'] += ['titanium']
                        hatch = 1
                    elif ti_sed:
                        data['extra'] += ['titanium']
                        sedan = 1
                    elif ti:
                        data['extra'] += ['titanium']
                    elif se_hat:
                        data['extra'] += ['se']
                        hatch = 1
                    elif se_sed:
                        data['extra'] += ['se']
                        sedan = 1
                    elif svt:
                        data['extra'] += ['svt']
                    elif sel:
                        data['extra'] += ['sel']
                    elif ses:
                        data['extra'] += ['se']
                        sedan = 1
                    elif se:
                        data['extra'] += ['se']
                    elif s_hat:
                        data['extra'] += ['s']
                        hatch = 1
                    elif s_sed:
                        data['extra'] += ['s']
                        sedan = 1
                    elif s:
                        data['extra'] += ['s']
                    else:
                        data['extra'] += [np.nan]


            else:
                data['year'] += [np.nan]
                data['extra'] += [np.nan]

            # Filling in other features
            seen = ['year','price', 'date', 'time','extra', 'model']

            for d in dat[1].findAll('span'):
                feat = d.get_text().split(': ')
                if feat[0] == 'cylinders':
                    cyl = re.search('[0-9]', feat[1])
                    if cyl:
                        data['cylinders'] += [int(cyl.group(0))]

                    else:
                        data['cylinders'] += [np.nan]
                    seen += ['cylinders']

                elif feat[0] == 'fuel':
                    if feat[1] == 'gas':
                        data['fuel'] += ['gas']
                    elif feat[1] == 'diesel':
                        data['fuel'] += ['diesel']
                    elif feat[1] == 'hybrid':
                        data['fuel'] += ['hybrid']
                    elif feat[1] == 'electric':
                        data['fuel'] += ['electric']
                    else:
                        data['fuel'] += ['other']
                    seen += ['fuel']

                elif feat[0] == 'odometer':
                    try:
                        data['odometer'] += [int(feat[1])]
                    except ValueError:
                        data['odometer'] += [np.nan]
                    seen += ['odometer']

                elif feat[0] == 'transmission':
                    if feat[1] == 'automatic':
                        data['transmission'] += ['auto']
                    elif feat[1] == 'manual':
                        data['transmission'] += ['manual']
                    else:
                        data['transmission'] += ['other']
                    seen += ['transmission']

                elif feat[0] == 'title status':
                    if feat[1] == 'clean':
                        data['title_stat'] += ['clean']
                    elif feat[1] =='salvage':
                        data['title_stat'] += ['salvage']
                    elif feat[1] =='rebuilt':
                        data['title_stat'] += ['rebuilt']
                    elif feat[1] =='parts only':
                        data['title_stat'] += ['parts']
                    elif feat[1] =='lien':
                        data['title_stat'] += ['lien']
                    else:
                        data['title_stat'] += ['missing']
                    seen += ['title_stat']

                elif feat[0] == 'type':
                    if hatch == 1 and sedan == 0 and coupe == 0:
                        data['type'] += ['hatchback']
                    elif hatch == 0 and sedan == 1 and coupe == 0:
                        data['type'] += ['sedan']
                    elif hatch == 1 and sedan == 0 and coupe == 1:
                        data['type'] += ['coupe']
                    elif feat[1] == 'coupe':
                        data['type'] += ['coupe']
                    elif feat[1] == 'hatchback':
                        data['type'] += ['hatchback']
                    elif feat[1] == 'sedan':
                        data['type'] += ['sedan']
                    else:
                        data['type'] += ['other']
                    seen += ['type']

                elif feat[0] == 'drive':
                    if feat[1] == 'fwd':
                        data['drive'] += ['fwd']
                    elif feat[1] == 'rwd':
                        data['drive'] += ['rwd']
                    else:
                        data['drive'] += ['4wd']
                    seen += ['drive']

                elif feat[0] == 'paint color':
                    data['color'] += [feat[1]]
                    seen += ['color']

                elif feat[0] == 'condition':
                    data['condition'] += [feat[1]]
                    seen += ['condition']

            # Filling in data not found
            keys = pd.Series(data.keys()[:])
            notseen = keys[np.logical_not(keys.isin(seen))]
            for ns in notseen:
                data[ns] += [np.nan]

            # Make dataframe
            testdf = pd.DataFrame(data)
            # testdf will be encoded, make a non-encoded version
            df_temp = testdf.copy() # Make a copy for reading off later

        # AUTOTRADER TIME
        else:
            soup2 = BeautifulSoup(html.text, 'lxml')

            # Add price, if no price - skip to next car
            price = soup2.find('span', {'title': re.compile('[pP]rice')})
            price_check = 0 # if price_check = 1, means null value
            if price == None:
                # FILTER OUT LATER BECAUSE OF RIDICULOUSNESS
                data['price'] = 0
                price_check = 1
            else:
                price = price.get_text().lstrip('$').replace(',', '')
                data['price'] += [int(price)]

            title = soup2.title.get_text()
            # Determine car model
            query = re.search('[cC][iI][vV][iI][cC]|' +\
                              '[cC][oO][rR][oO][lL][lL][aA]|' +\
                              '[fF][oO][cC][uU][sS]|' +\
                              '[cC][rR][uU][zZ][eE]|' +\
                              '[eE][lL][aA][nN][tT][rR][aA]|' +\
                              '[fF][oO][rR][tT][eE]|' +\
                              '[mM][aA][zZ][dD][aA]', title)

            # Making sure we get SOMETHING
            if query:
                query = query.group(0).lower()
            else:
                reason = 'Did not detect civic, corolla, focus, cruze' +\
                         ', elantra, forte, or mazda'
                return render_template("invalid.html", reason = reason)

            # Add model
            if query == 'civic':
                data['model'] += ['civic']
            elif query == 'corolla':
                data['model'] += ['corolla']
            elif query == 'focus':
                data['model'] += ['focus']
            elif query == 'cruze':
                data['model'] += ['cruze']
            elif query == 'elantra':
                data['model'] += ['elantra']
            elif query == 'mazda':
                data['model'] += ['mazda3']
            elif query == 'forte':
                data['model'] += ['forte']

            # Add highway mpg
            hwympg = soup2.find('div', {'class': re.compile('mpg-hwy')})
            if hwympg != None:
                hwympg = hwympg.span.get_text()
                try:
                    data['hwympg'] += [int(hwympg)]
                except:
                    data['hwympg'] += [np.nan]
            else:
                data['hwympg'] += [np.nan]

            # Add city mpg
            citympg = soup2.find('div', {'class': re.compile('mpg-city')})
            if citympg != None:
                citympg = citympg.span.get_text()
                try:
                    data['citympg'] += [int(citympg)]
                except:
                    data['citympg'] += [np.nan]
            else:
                data['citympg'] += [np.nan]

            # Add fuel
            try:
                fuel = soup2.findAll('span', {'class': 'mpg'})[2]
            except:
                fuel = None
            if fuel != None:
                fuel = fuel.get_text()
                try:
                    data['fuel'] += [fuel]
                except:
                    data['fuel'] += ['unknown_fuel']
            else:
                data['fuel'] += ['unknown_fuel']

            # Add year
            title = soup2.title.get_text()
            year = re.search('([^a-zA-Z0-9:\.*_]\s*[12][09][0-9][0-9]'+\
                              '[^a-zA-Z0-9~_\-\.]|'+\
                              '^[12][09][0-9][0-9][^a-zA-Z0-9~_\-\.])',
                              title)
            if year != None:
                try:
                    data['year'] += [int(year.group(0))]
                except:
                    reason = 'No reported year'
                    return render_template("invalid.html", reason = reason)
            else:
                reason = 'No reported year'
                return render_template("invalid.html", reason = reason)

            # Add condition
            condition = soup2.find('h1',
                                  {'class': 'listing-title atcui-block'})
            if condition != None:
                try:
                    condition = condition.get_text().split(' ')
                    if condition[0] == 'Certified':
                        data['condition'] += ['certified']
                    elif condition[0] == 'New':
                        data['condition'] += ['new']
                    elif condition[0] == 'Used':
                        data['condition'] += ['used']
                    else:
                        data['condition'] += ['unknown_condition']
                except:
                    data['condition'] += ['unknown_condition']
            else:
                data['condition'] += ['unknown_condition']

            # Add color
            color = soup2.find('span', {'class':"colorName"})
            if re.search('[eE]xterior', color.get_text()):
                color = color.span.get_text()
                data['color'] += [color]
            else:
                data['color'] += ['unknown_color']

            # Add cylinders
            cyl = re.search('[0-9] [cC]ylinder', soup2.get_text())
            if cyl:
                cyl = cyl.group(0).split(' ')
                data['cylinders'] += [int(cyl[0])]
            else:
                data['cylinders'] += [np.nan]

            # Add extra and type
            trim = soup2.find('span', {'class':'heading-trim'})
            if trim:
                trim = trim.get_text().split(' ')
                data['extra'] += [trim[1]]
                if len(trim) > 2:
                    data['type'] += [trim[2]]
                else:
                    data['type'] += [np.nan]
            else:
                data['extra'] += [np.nan]
                data['type'] += [np.nan]

            # Add odometer
            odometer = soup2.find('span', {'class':"atcui-clear heading-mileage"})
            if odometer != None:
                odometer = odometer.get_text().split(' ')
                odometer = odometer[0]
                odometer = odometer.replace(',', '')
                data['odometer'] += [int(odometer)]
            else:
                data['odometer'] += [np.nan]

            # Make dataframe
            testdf = pd.DataFrame(data)
            # testdf will be encoded, make a non-encoded version
            df_temp = testdf.copy() # Make a copy for reading off later





        #####################
        # GRAB DATA FOR LABELENCODING
        ######################







        # Connect with MySQL and grab data into one dataframe
        connect = dbConnect(host = 'localhost', user = 'root',
                            passwd = 'default', db = 'find_car')

        with connect:
            # Grab dataframes from SQL and concatenate them into one
            if site == 'craigslist':
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

                df = pd.concat(df_dict.values(), axis = 0)
                df = df.reset_index().drop('index', axis = 1)

                # Filter bad titles and conditions
                df = df[df['title_stat'] != 'salvage']
                df = df[df['title_stat'] != 'missing']
                df = df[df['title_stat'] != 'lien']
                df = df[df['condition'] != 'salvage']

                # Filter ridiculous prices
                df = df[df['price'] > 1000]

                # Correct for null values
                for col in df.columns:
                    # Honda civics are 4 cylinders
                    if col == 'cylinders':
                        df.loc[pd.isnull(df[col]), col] = 4
                        testdf.loc[pd.isnull(testdf[col]), col] = 4
                    elif col == 'odometer':
                        # From federal highway admin on avg number of miles/year
                        now_year = int(datetime.datetime.now().year)
                        df.loc[pd.isnull(df[col]), col] = \
                               13476*(now_year - data['year'][0])*2
                        testdf.loc[pd.isnull(testdf[col]), col] = \
                               13476*(now_year - data['year'][0])*2
                    elif col == 'extra':
                        df.loc[pd.isnull(df[col]), col] = 'unknown'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'unknown'
                    elif col == 'fuel':
                        df.loc[pd.isnull(df[col]), col] = 'gas'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'gas'
                    elif col == 'drive':
                        df.loc[pd.isnull(df[col]), col] = 'fwd'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'fwd'
                    elif col == 'transmission':
                        df.loc[pd.isnull(df[col]), col] = 'auto'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'auto'
                    elif col == 'type':
                        df.loc[pd.isnull(df[col]), col] = 'unknown'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'unknown'
                    elif col == 'color':
                        df.loc[pd.isnull(df[col]), col] = 'unknown'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'unknown'
                    elif col == 'condition':
                        df.loc[pd.isnull(df[col]), col] = 'unknown'
                        testdf.loc[pd.isnull(testdf[col]), col] = 'unknown'
                    elif col == 'date':
                        df[col] = pd.to_datetime(df.date, format = '%Y-%m-%d')
                        testdf[col] = pd.to_datetime(testdf.date,
                                                    format = '%Y-%m-%d')

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
                df = pd.concat(df_dict.values(), axis = 0)
                df = df.reset_index().drop('index', axis = 1)

                # For NULL values, average over non_null values for each model
                # Replace NULL values with average value - not much std
                for col in ['citympg', 'hwympg', 'cylinders']:
                    for val in df['model'].unique():
                        mean = np.mean(df.loc[(pd.notnull(df[col])) \
                                       & (df['model'] == val), col])

                        df.loc[(pd.isnull(df[col])) \
                                    & (df['model'] == val),col] = mean
                        testdf.loc[(pd.isnull(testdf[col])) \
                                    & (testdf['model'] == val),col] = mean





        if site == 'craigslist':
            columns = ['color', 'condition', 'drive', 'extra', 'fuel', 'model',
                       'odometer', 'title_stat', 'transmission', 'type']
        elif site == 'autotrader':
            columns = ['model', 'condition', 'extra', 'type', 'color', 'fuel']

        # Convert date column
        if site == 'craigslist':
            min_date = df['date'].min()
            df['date'] = (df['date'] - min_date)
            df['date'] = df['date'].astype(str)

            testdf['date'] = (testdf['date'] - min_date)
            testdf['date'] = testdf['date'].astype(str)
            def date_convert(x):
                dates = x.split(' ')
                return dates[0]
            df['date'] = (df['date'].apply(date_convert)).astype(int)
            testdf['date'] = (testdf['date'].apply(date_convert)).astype(int)

        for col in columns:
            if type(df[col].unique()[1]) == str:
                le = PruneLabelEncoder()
                le.fit(df[col], TRANSFORM_CUTOFF)
                # Label Encode new DF
                print col
                print testdf[col]
                try:
                    testdf[col] = le.transform(testdf[col])
                except:
                    reason = 'column encode: ' + col + ' = '+str(testdf[col][0])
                    return render_template("invalid.html", reason = reason)

        # Drop null values in year
        # Do linear regression prediction later for improved feature
        df = df[pd.notnull(df['year'])]
        if testdf[pd.notnull(testdf['year'])].shape[0] == 0:
            reason = 'No reported year'
            return render_template("invalid.html", reason = reason)

        # Place price column at the end
        price = df['price'].copy()
        df = df.drop(['price'], axis = 1)
        df['price'] = price

        price2 = testdf['price'].copy()
        testdf = testdf.drop(['price'], axis = 1)
        testdf['price'] = price2

        print 'input data:'

        print testdf
        print
        print df_temp

        #####################
        # BRING IN TRAINED MODEL
        ######################

        # Use pickle to load the model trained earlier
        # Model is site dependent
        if site == 'craigslist':
            filename = 'model/cg_model.pickle'
        else:
            filename = 'model/at_model.pickle'

        X = testdf.as_matrix(testdf.columns[:-1])
        with open(filename,'r') as f:
            mdl = pickle.load(f)
        pred = mdl.predict(X)
        price = testdf['price'][0]
        print 'predicted price', pred[0]
        print 'real price: ', price

        # Calculate if this is a deal
        deal = pred[0] - price
        pct = str(round(float(deal)/price*100,1))+"%"
        print 'percent:', pct
        deal = round(deal, 0)
        deal_compare = deal

        gooddeal = 0
        color_text = "#F78181"
        if price_check == 1:
            reason = 'NO LISTED PRICE!'
            return render_template("invalid.html", reason = reason)
        elif deal > 0:
            suggestion = 'Good Deal'
            suggestion1 = 'Savings'
            suggestion2 = ''
            gooddeal = 1 # Used to toggle Green on web page for good deal
            deal = '$' + str(deal)
            color_text = "#72B095"
        else:
            suggestion = 'Bad Deal'
            suggestion1 = ''
            suggestion2 = 'Overpaying by'
            gooddeal = 0 # Used to toggle Red on web page for bad deal
            deal = str(deal).split('-')
            #deal[0] = '-'
            deal = '$' + deal[1]
            # color_text = "#D94639"
            # color_text = "#F78181"
            color_text = "#8A0808"

        price = round(price, 0)
        pred = round(pred[0], 0)
        model = df_temp['model'][0]
        odometer = df_temp['odometer'][0]
        year = df_temp['year'][0]

        # Set numbers to make for Highcharts graph
        price_num = price
        pred_num = pred
        max_num = round(max(price_num, pred_num)*1.8, 0)
        min_num = round(min(price_num, pred_num)*0.95, 0)


        # Convert df_temp to a dictionary to loop through for webpage
        entries = dict(model = model,
                       deal = deal,
                       price = "${:,.0f}".format(price),
                       prediction = "${:,.0f}".format(pred),
                       year = year,
                       odometer = "{:,.0f}".format(odometer))

        ########################################################################
        # Get Recommendations for Cars
        ########################################################################
        # Connect with MySQL and grab data into one dataframe
        connect = dbConnect(host = 'localhost', user = 'root',
                            passwd = 'default', db = 'find_car')

        with connect:
            if site == 'craigslist':
                sql = 'SELECT * FROM result_cg;'
                df_result = pd.read_sql_query(sql,
                                              con = connect.con,
                                              index_col = 'index')
            else:
                sql = 'SELECT * FROM result_at;'
                df_result = pd.read_sql_query(sql,
                                              con = connect.con,
                                              index_col = 'index')

        # Define Deal Quality
        numerator = df_result[df_result['deal'] < deal_compare].shape[0]
        denominator = df_result.shape[0]
        dealquality = float(numerator)/float(denominator) * 100
        dealquality = round(dealquality, 1)

        # Filter SQL query to get recommendations
        df_result = df_result[df_result['model'] == model]
        if df_result.shape[0] > 0:
            df_result = df_result[(df_result['price'] < 1.2*price) & \
                                  (df_result['price'] > 0.8*price)]
            if df_result.shape[0] > 0:
                df_result = df_result[(df_result['year'] < 1.4*price) & \
                                      (df_result['year'] > 0.6*year)]
                if df_result.shape[0] > 0:
                    df_result = df_result[(df_result['deal'] > deal_compare) & \
                                          (df_result['deal'] < 7000) & \
                                          (df_result['deal'] > 100)]
        df_result = df_result.sort_values('deal', axis = 0, ascending = False)

        # If many results, get top 3 results
        #if df_result.shape[0] > 9:
        #    df_result = df_result.iloc[:10, :]

        df_result = df_result.reset_index().drop('index', axis = 1)

        # Format results for HTML
        rec = [dict(model = df_result['model'][row],
               deal = "${:,.0f}".format(df_result['deal'][row]),
               price = "${:,.0f}".format(df_result['price'][row]),
               prediction = "${:,.0f}".format(df_result['pred'][row]),
               year = df_result['year'][row],
               odometer = "{:,.0f}".format(df_result['odometer'][row]),
               url = df_result['url'][row]) \
               for row in range(df_result.shape[0])]

        """
        # Debugging
        output_file("tt2.html")
        f = open('tfile', 'w')
        f.write(plot_script)
        f.close()
        show(p)
        """
        # print suggestion, gooddeal
        # print gooddeal == 1
        print "${:,.0f}".format(df_result['pred'][0])
        if site == 'craigslist':
            html = render_template("carcheck.html", entries = entries,
                                   url = url, suggestion1 = suggestion1,
                                   suggestion2 = suggestion2, rec = rec,
                                   gooddeal = gooddeal,
                                   dealquality = dealquality, site = site,
                                   price = price_num, pred = pred_num,
                                   max_num = max_num, min_num = min_num,
                                   color = color_text, pct = pct,
                                   suggestion = suggestion)
        else:
            html = render_template("carcheck.html", entries = entries,
                                   url = url, suggestion1 = suggestion1,
                                   suggestion2 = suggestion2, rec = rec,
                                   gooddeal = gooddeal,
                                   dealquality = dealquality, site = site,
                                   price = price_num, pred = pred_num,
                                   max_num = max_num, min_num = min_num,
                                   color = color_text, pct = pct,
                                   suggestion = suggestion)

    else:
        # It will be a get request instead
        url = "http://sfbay.craigslist.org/pen/ctd/5413262011.html"
        html = render_template("carcheckblank.html", url = url)

    return html
