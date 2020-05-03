#!/usr/bin/env python
# coding: utf-8

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from string import Template
import pandas as pd

process = CrawlerProcess({'AUTOTHROTTLE_ENABLED': True, 
                          'AUTOTHROTTLE_TARGET_CONCURRENCY': .20,
                          'HTTPCACHE_ENABLED': True,  # remove for final scrape to get live data
                          'ROBOTSTXT_OBEY': True,
                          'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36',
                          'FEED_URI': '../data/ots_snowfall_data.json',
                          'FEED_FORMAT': 'json',
                          'FEED_EXPORT_ENCODING': '"utf-8"'})

class OpenSnowSnowfallSpider(scrapy.Spider):
    name = 'opensnow'
    
    links_df = pd.read_feather('../data/ots_station_links.feather')  # remove for full run
    # load values and remove nested listing
    start_urls = [x[0] for x in links_df.filter(items=['snow_link']).values.tolist()]

    def parse(self, response, links_df=links_df):
        month_xp = ".//div[starts-with(@class, 'cal_chart_di')]"
        month_name_xp = ".//strong[@class='dte_mon']/text()"
        date_xp = ".//span[@class='dte_hd']"
        data_xp = "../div/text()"
        data = {}
        for month in response.xpath(month_xp):
            month_name = month.xpath(month_name_xp).get()
            data[month_name] = {}
            for day in month.xpath(date_xp):
                date = day.xpath("./text()").get()
                day_data = day.xpath(data_xp).get()
                # only save non-zero data. the later data cleaning step will use pd features
                # to do datetime cleaning/interpolation
                if date == r"\xa0" or day_data == "" or date is None or day_data is None:
                    continue
                data[month_name][date] = int(day_data)
        source_data = links_df.loc[links_df.snow_link == response.request.url, :]
        to_save= {'station':  source_data.resort_name_short.values[0],
               'what_data': source_data.page.values[0],
               'year' : int(source_data.year.values[0]),  # for typeerror durin json serialization
               'url': response.request.url,
               'data': data,
               }
        #print(f'HERE IS THE DATA -----{to_save}----------')
        yield to_save

process.crawl(OpenSnowSnowfallSpider)
process.start()