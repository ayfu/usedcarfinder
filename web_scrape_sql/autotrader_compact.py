'''

__file__

    autotrader_compact.py

__description__

    This file provides a class to scrape autotrader for features and labels on
    used compact cars.

'''
import re
import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


class ScrapeAutotrader():
    def __init__(self, zipcode = 94306, maxprice = 500000, make = 'Honda',
                 model = 'Civic', radius = 50, startyear = 1981):
        self.zipcode = zipcode
        self.maxprice = maxprice
        self.make = make
        self.model = model
        self.radius = radius
        self.startyear = startyear

        # Set first_link url
        self.url = 'http://www.autotrader.com/cars-for-sale/Used+Cars/'+\
                    'cars+under+'+str(maxprice)+'/'+self.make+'/'+self.model +\
                   'Palo+Alto-'+str(zipcode)
        parameters = ['?endYear=2017',
                      'firstRecord=0',
                      'makeCode1='+self.make.upper(),
                      'maxPrice='+str(self.maxprice),
                      'modelCode1='+self.model.upper(),
                      'searchRadius='+str(self.radius),
                      'showcaseOwnerId=54471071',
                      'startYear='+str(self.startyear),
                     ]
        self.url = self.url + '&'.join(parameters)

    def page_data(self, url):
        '''
        Takes in: url

        Returns: dataframe scraped from the Autotrader web page

        I also found that cylinder, drive type, and fuel don't really
        make a difference for these cars
        '''

        hdr = {'Host': 'www.autotrader.com',
               'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; ' +\
                             'rv:43.0) Gecko/20100101 Firefox/43.0'
               }
        data = {'url': [],
                'model': [],
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

        r = requests.get(url, headers = hdr)
        soup = BeautifulSoup(r.content, "lxml")
        cars = soup.findAll('div', {'class':re.compile('listing-isClickable')})

        for car in cars:
            url = car.find('a', {'href':re.compile('cars-for-sale')})
            if url != None:
                url = url.attrs['href']
                url = 'http://www.autotrader.com' + url
                html = requests.get(url, headers = hdr)
                soup2 = BeautifulSoup(html.text, "lxml")

                # Add price, if no price - skip to next car
                price = soup2.find('span', {'title': re.compile('[pP]rice')})
                if price != None:
                    price = price.get_text().lstrip('$').replace(',', '')
                    data['price'] += [int(price)]
                else:
                    continue

                # Add url
                data['url'] += [url]

                # Add model
                if self.model == 'Civic':
                    data['model'] += ['civic']
                elif self.model == 'Corol':
                    data['model'] += ['corolla']
                elif self.model == 'Focus':
                    data['model'] += ['focus']
                elif self.model == 'Cruze':
                    data['model'] += ['cruze']
                elif self.model == 'Elantr':
                    data['model'] += ['elantra']
                elif self.model == 'Mazda3':
                    data['model'] += ['mazda3']
                elif self.model == 'Forte':
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
                        data['year'] += [np.nan]
                else:
                    data['year'] += [np.nan]

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

            else:
                continue

        # Find the link for the next page
        link = soup.find('a',
                        {'class':'pagination-button pagination-button-next'})
        if link:
            link = link.parent.find('a',
                                    {'href': re.compile('/cars-for-sale/.*')})
            self.nextlink = 'http://www.autotrader.com'+link.attrs['href']
        else:
            self.nextlink = None

        # Make dataframe from data dictionary
        df = pd.DataFrame(data)
        return df

    def all_data(self, url):
        """
        Takes in: Starting Autotrader url for cars (for civics)

        Returns: Full dataframe from scraping all pages until the end

        """
        # Create dictionary of dataframes
        df_dict = {}
        i = 1
        df_dict['page'+str(i)] = self.page_data(url)
        print 'page ' + str(i) + ' scraped'
        # Keep scraping webpages until there is no "Next" button
        while self.nextlink != None:
            i += 1
            name = 'page'+str(i)
            df_temp = self.page_data(self.nextlink)
            df_dict[name] = df_temp
            print 'page ' + str(i) + ' scraped'

        self.df = pd.concat(df_dict.values(), axis = 0)
        self.df = self.df.reset_index().drop('index', axis = 1)

        # Clean null values from columns
        for col in self.df.columns:
            if col == 'color':
                self.df.loc[pd.isnull(self.df[col]),col] = 'unknown'
                self.df.loc[self.df[col] == '\n', col] = 'unknown'
            elif col == 'condition':
                self.df.loc[pd.isnull(self.df[col]),col] = 'unknown'
                self.df.loc[self.df[col] == '\n', col] = 'unknown'
            elif col == 'distance':
                self.df = self.df[pd.notnull(self.df[col])]
            elif col == 'extra':
                self.df.loc[pd.isnull(self.df[col]),col] = 'unknown'
                self.df.loc[self.df[col] == '\n', col] = 'unknown'
            elif col == 'odometer':
                self.df = self.df[pd.notnull(self.df[col])]
            elif col == 'type':
                self.df.loc[pd.isnull(self.df[col]),col] = 'unknown'
                self.df.loc[self.df[col] == '\n', col] = 'unknown'
            elif col == 'year':
                self.df = self.df[pd.notnull(self.df[col])]

        df = self.df.copy()
        return df
