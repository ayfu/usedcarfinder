import sqlite3
import os
from contextlib import closing
from collections import OrderedDict
import re
from collections import defaultdict
import pickle

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

# set site = 'craigslist' right now
site = 'craigslist'
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
    return render_template("index.html",
                            title = 'Home',
                            user = user)





@app.route('/carcheck', methods = ['GET', 'POST'])
def carcheck():

    #ALL = request.args.get('ALL')
    # Monitor if url is good
    bad_url = 0

    if request.method == 'POST':
        urlcall = request.form['urlcall']
        url = urlcall.lower()
    else:
        url = 'http://sfbay.craigslist.org/pen/ctd/5413262011.html'

    if request.method == 'POST':
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
        except:
            return render_template("invalid.html")
        soup = BeautifulSoup(html.text, 'lxml')
        if soup:
            dat = soup.findAll('p', attrs={'class':"attrgroup"})
        else:
            url = 'http://sfbay.craigslist.org/pen/ctd/5413262011.html'
            html = requests.get(url)
            soup = BeautifulSoup(html.text, 'lxml')

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
                          '[mM][aA][zZ][dD][aA]', title).group(0)

        # Making sure we get SOMETHING
        if query == None:
            return render_template("invalid.html")
        query = query.lower()

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
                data['year'] += [np.nan]

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

        # Correct for null values
        for col in df.columns:
            # Honda civics are 4 cylinders
            if col == 'cylinders':
                df.loc[pd.isnull(df[col]), col] = 4
                testdf.loc[pd.isnull(testdf[col]), col] = 4
            elif col == 'odometer':
                df.loc[pd.isnull(df[col]), col] = 20000
                testdf.loc[pd.isnull(testdf[col]), col] = 20000
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
                testdf[col] = pd.to_datetime(testdf.date, format = '%Y-%m-%d')



        if site == 'craigslist':
            columns = ['color', 'condition', 'drive', 'extra', 'fuel', 'model',
                       'odometer', 'title_stat', 'transmission', 'type']
        elif site == 'autotrader':
            columns = ['color', 'condition']

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
                testdf[col] = le.transform(testdf[col])

        # Drop null values in year
        # Do linear regression prediction later for improved feature
        df = df[pd.notnull(df['year'])]
        if testdf[pd.notnull(testdf['year'])].shape[0] == 0:
            # Come up with something smarter
            testdf['year'] = 2000

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
        deal = round(deal, 2)

        if price_check == 1:
            suggestion = 'NO LISTED PRICE! RUN AWAY FROM THIS SNEAKY PERSON'
        elif deal > 0:
            suggestion = 'GOOD DEAL'
        else:
            suggestion = 'BAD DEAL'

        deal = str(deal) + ' Dollars'



        # convert input_team to a string for html and then passed to img function

        """
        # Debugging
        output_file("tt2.html")
        f = open('tfile', 'w')
        f.write(plot_script)
        f.close()
        show(p)
        """
        print suggestion
        html = render_template("carcheck.html", deal = deal,
                                url = url,
                                suggestion = suggestion,
                                model = df_temp['model'][0],
                                year = df_temp['year'][0],
                                odometer = df_temp['odometer'][0],
                                date = df_temp['date'][0])

    else:
        # It will be a get request instead
        url = "http://sfbay.craigslist.org/pen/ctd/5413262011.html"
        html = render_template("carcheck.html", url = url)

    return html


# Not using anymore

@app.route('/img/<input_team>')
def img(input_team):
    input_team = list(input_team.split(','))
    print len(input_team)
    print input_team
    sql = "SELECT * FROM results"
    df = pd.read_sql_query(sql, g.db)
    df = df.drop('index', axis = 1)
    print 'Dataframe shape:', df.shape
    print list(df.columns)
    #con = connect_db()
    #cur = con.cursor()
    #cur.execute("SELECT * FROM results;")
    #con.commit()
    #print cur.fetchall()
    #con.close()
    p_df = df[df['team'].isin(input_team)].copy()
    print 'New Dataframe shape:', p_df.shape
    pred = np.array(p_df['pred'])
    y_test = np.array(p_df['points'])
    avg_pm = np.array(p_df['avg_pm'])


    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize = (10, 8))

    x1 = np.arange(0,180)
    y1 = np.zeros(180)+180
    x2 = np.arange(0,-180, -1)
    y2 = np.zeros(180)-180
    ax.plot(x1, y1)
    plt.fill_between(x1, y1, 0, color=(0.01,0.40,0.1), alpha = 0.25)
    plt.fill_between(x2, y2, 0, color=(0.01,0.40,0.1), alpha = 0.25)
    ax.scatter(y_test, avg_pm, color = (0,0.2,0.5),
               label = 'Base Model Predictions', s = 70, alpha = 1)
    ax.scatter(y_test, pred, color = (0.6,0.0,0.2),
               label = 'New Model Predictions',
               s = 70, alpha = 1)
    ax.plot(np.arange(-200, 200),np.arange(-200, 200), color = 'black',
               label = 'Perfect Prediction Line',
               lw = 3, alpha = 0.6, ls = 'dashed')
    #ax.plot(x,pred_y, label = 'Fit', lw = 5)
    ax.set_xlabel('Actual +/- (points/48 min)',fontsize = 14)
    ax.set_ylabel('Predicted +/- (points/48 min)', fontsize = 14)
    ax.set_title('Prediction Results', fontsize = 20)
    ax.set_xlim(-175,175)
    ax.set_ylim(-175,175)
    ax.legend(loc=2, fontsize = 12)
    ax.tick_params(labelsize =12)

    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response
