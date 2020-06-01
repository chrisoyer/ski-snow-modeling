#!/usr/bin/env python
# coding: utf-8

# This pulls a list of ski resorts from On The Snow. This is used to generate a list of pages to scrap for snow data.
import numpy as np
import pandas as pd
import json
import urllib.request

## Pull data
limit = 500
source_url = (r'https://skiapp.onthesnow.com/app/widgets/resortlist?region=us'
              r'&regionids=429&language=en&pagetype=skireport&direction=-1'
              fr'&order=stop&limit={limit}&offset=30&countrycode=USA'
              r'&minvalue=-1&open=anystatus')

with urllib.request.urlopen(source_url) as url_file:
    station_data = url_file.read().decode()
    json_data = json.loads(station_data)

# json -> dataframe
station_df = pd.DataFrame.from_dict(json_data['rows'])
station_data_df = (station_df
                   .join(pd.DataFrame.from_dict(
                       station_df[['links']].to_dict()['links'])
                         .T)
                   .filter(items=['resort_name_short', 'weather'])
                   )

# if we get as many station as requested, there could be more we didn't get
assert limit > station_data_df.shape[0]
pages = {'snowfall': "", 'base': "&q=top"}

def link_fixer(ser, year, page):
    """creates correct links from relative links for different page."""
    base = 'https://www.onthesnow.com'
    tail = f'historical-snowfall.html?y={year}'
    ser2 = ser.str.replace(pat=r'weather.html', repl=tail)
    return base + ser2 + pages[page]

def link_maker(df, yr, page):
    """makes df based on year given"""
    return (df
            .assign(snow_link=link_fixer(df.weather, yr, page))
            .assign(year=yr)
            .assign(page=page)
            )

years = range(2010, 2019)
link_dfs = [link_maker(station_data_df, yr, page) 
            for yr in years for page in pages.keys()]
station_links_df = pd.concat(link_dfs).drop(columns='weather')

# save to disk
(station_links_df
.reset_index(drop=True)
.to_feather('../data/ots_station_links.feather'))