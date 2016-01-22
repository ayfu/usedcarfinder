'''

__file__

    cglst_compact.py

__description__

    This file provides a class to scrape craigslist for features and labels on
    used compact cars. The inputs are min price, max price, min mileage,
    and max mileage

'''
import re
import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

class ScrapeCglst():
    def __init__(self, minprice = 0, maxprice = 200000,
                 model = 'Honda+Civic', min_year = 0,
                 max_year = 2017, min_miles = 0, max_miles = 500000):
        self.minprice = minprice
        self.maxprice = maxprice
        self.model = model
        self.min_year = min_year
        self.max_year = max_year
        self.min_miles = min_miles
        self.max_miles = max_miles
        self.query = model.lower()

        #Set first_link url
        self.url = "http://sfbay.craigslist.org/search/cta"
        parameters = ['?min_price='+str(self.minprice),
                      'max_price='+str(self.maxprice),
                      'auto_make_model='+self.model,
                      'min_auto_year='+str(self.min_year),
                      'max_auto_year='+str(self.max_year),
                      'min_auto_miles='+str(self.min_miles),
                      'max_auto_miles='+str(self.max_miles),
                      'query='+str(self.query)]
        self.url = self.url+'&'.join(parameters)

    def page_data(self, url):
        """
        Input: Craigslist url for cars

        Returns: dataframe of car data for a single page
        """
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'lxml')
        cars = soup.findAll('p', attrs={'class': 'row'})

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

        for car in cars:
            url2 = "http://sfbay.craigslist.org/" + str(car.a['href'])
            html2 = requests.get(url2)
            soup2 = BeautifulSoup(html2.text, 'lxml')
            dat = soup2.findAll('p', attrs={'class':"attrgroup"})

            # Take care of price, time, and date first
            # try statement for one anomolous page for Mazda 3
            try:
                time = soup2.find('time',
                                attrs={'datetime': re.compile('.*')}).get_text()
            except:
                continue
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

            price = car.find('span', attrs={'class': 'price'})
            if price == None:
                # FILTER OUT LATER BECAUSE OF RIDICULOUSNESS
                data['price'] = 0
            else:
                price = price.get_text()
                data['price'].append(float(price.split('$')[-1]))


            # Checks if title contains car type (ie. hatchback)
            hatch = 0
            sedan = 0
            coupe = 0

            # Fill in first features (from webpage or from title)
            if dat[0].span.get_text() != None:
                title = soup2.title.get_text()
                # year is from title
                year = dat[0].span.get_text()
                year = re.search('([^a-zA-Z0-9\.*_]\b[12][09][0-9][0-9]'+\
                                 '[^a-zA-Z0-9~_\-\.]|'+\
                                 '^[12][09][0-9][0-9][^a-zA-Z0-9~_\-\.])',
                                 title)
                if year:
                    data['year'] += [int(year.group(0))]
                else:
                    data['year'] += [np.nan]

                # Different Cars - scrape 'extra' feature from title

                # CIVIC
                if self.query == 'honda+civic':
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
                if self.query == 'toyota+corolla':
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
                if self.query == 'chevrolet+cruze':
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
                if self.query == 'hyundai+elantra':
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
                if self.query == 'mazda+3':
                    data['model'] += ['mazda3']
                    data['extra'] += [np.nan]

                # KIA FORTE
                if self.query == 'kia+forte':
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
                if self.query == 'ford+focus':
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

            # Reset variables that check if car type can be extracted from title
            hatch = 0
            sedan = 0
            coupe = 0

        #Grab the link to the next page
        next_link = soup.find('link', attrs={"rel":"next"})
        if next_link:
            self.nextlink = next_link['href']
        else:
            self.nextlink = None

        #Make dataframe
        df = pd.DataFrame(data)
        return df

    def all_data(self, url):
        """
        Takes in: starting craigslist url for cars (Specific for honda civic)

        Returns: a dataframe from all the page listing of Honda Civics on
                 craigslist (for given parameters)
        """
        df_dict = {} #dictionary of dataframes
        i = 1
        df_dict['page'+str(i)] = self.page_data(url)
        print 'page ' + str(i) +' scraped'
        while self.nextlink != None:
            i += 1
            print 'page ' + str(i) +' scraped'
            name = 'page'+str(i)
            df_temp = self.page_data(self.nextlink)
            df_dict[name] = df_temp

        self.df = pd.concat(df_dict.values(), axis = 0)
        self.df = self.df.reset_index().drop('index', axis = 1)

        for col in self.df.columns:
            # Honda civics are 4 cylinders
            if col == 'cylinders':
                self.df.loc[pd.isnull(self.df[col]), col] = 4
            elif col == 'extra':
                self.df.loc[pd.isnull(self.df[col]), col] = 'unknown'
            elif col == 'fuel':
                self.df.loc[pd.isnull(self.df[col]), col] = 'gas'
            elif col == 'drive':
                self.df.loc[pd.isnull(self.df[col]), col] = 'fwd'
            elif col == 'transmission':
                self.df.loc[pd.isnull(self.df[col]), col] = 'auto'
            elif col == 'type':
                self.df.loc[pd.isnull(self.df[col]), col] = 'unknown'
            elif col == 'color':
                self.df.loc[pd.isnull(self.df[col]), col] = 'unknown'
            elif col == 'condition':
                self.df.loc[pd.isnull(self.df[col]), col] = 'unknown'
            elif col == 'date':
                self.df[col] = pd.to_datetime(self.df.date, format = '%Y-%m-%d')
            #elif col == 'time':
            #    self.df[col] = pd.to_datetime(self.df[col])
            #    self.df[col] = self.df[col].dt.time
            elif col == 'odometer':
                #No null right now, but we can avg over cars with same year
                #Future reference
                self.df.loc[pd.isnull(self.df[col]), col] = 0


        df = self.df.copy()
        return df
