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
        data = {'model': [],
                'condition': [],
                'year': [],
                'extra': [],
                'type': [],
                'odometer': [],
                'color': [],
                'price': [],
                'distance': []
                #'description': []
               }

        r = requests.get(url, headers = hdr)
        soup = BeautifulSoup(r.content, "lxml")
        cars = soup.findAll('div', {'class':re.compile('listing-isClickable')})

        for car in cars:

            title = car.find('span',{'class':'atcui-truncate ymm'})
            if title:

                # Check car type
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


                title = title.get_text().split(' ')
                if title[0] == 'Certified':
                    data['condition'] += ['certified']
                elif title[0] == 'New':
                    data['condition'] += ['new']
                elif title[0] == 'Used':
                    data['condition'] += ['used']
                else:
                    data['condition'] += ['unknown']

                data['year'] += [int(title[1])]


                trim = car.find('span',{'class':'trim'})
                if trim:
                    trim = trim.get_text().split(' ')
                    data['extra'] += [trim[0]]
                    if len(trim) > 1:
                        data['type'] += [trim[1]]
                    else:
                        data['type'] += [np.nan]
                else:
                    data['extra'] += [np.nan]
                    data['type'] += [np.nan]

                mileage = car.find('span',{'class':'mileage'})
                if mileage:
                    mileage = mileage.get_text().split(' ')
                    odo = mileage[0]
                    odo = odo.replace(',', '')
                    data['odometer'] += [int(odo)]
                else:
                    data['odometer'] += [np.nan]

                color = car.find('span',{'class':'atcui-block'})
                if color:
                    data['color'] += [color.get_text()]
                else:
                    data['color'] += [np.nan]

                price = car.find('h4',{'class':'primary-price'})
                if price:
                    price = price.get_text().lstrip('$').replace(',', '')
                    data['price'] += [int(price)]
                else:
                    data['price'] += [np.nan]

                dist = car.find('span',{'class':'distance-cont'})
                if dist:
                    dist = dist.get_text().split(' ')
                    data['distance'] += [int(dist[0])]
                else:
                    data['distance'] += [np.nan]
                """
                desc = car.find('span',{'class':'atcui-truncate atcui-block'})
                if desc:
                    desc = desc.get_text()
                    data['description'] += [desc]
                else:
                    data['description'] += [np.nan]
                """

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
