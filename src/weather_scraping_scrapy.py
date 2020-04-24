#!/usr/bin/env python
# coding: utf-8

# In[3]:


import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.shell import inspect_response
from string import Template


# xtra urls
na_ski_url = r"https://en.wikipedia.org/wiki/List_of_ski_areas_and_resorts_in_the_United_States"

process = CrawlerProcess({'AUTOTHROTTLE_ENABLED': True,  # or download delay
                          'HTTPCACHE_ENABLED': True,  # remove for final scrape to get live data
                          'ROBOTSTXT_OBEY': True,
                          'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36',
                          'FEED_URI': 'ski_data.json',
                          'FEED_FORMAT': 'json',
                          'FEED_EXPORT_ENCODING': '"utf-8"'})


class OpenSnowSpider(scrapy.Spider):
    name = 'opensnow'
    #opensnow_states = ['colorado', 'california', 'washington']
    #state_dict = {'colorado':}
    #base_resort_url = fr"https://www.onthesnow.com/{state}/{station}/historical-snowfall.html?y={year}&q=top

    def start_requests(self):
        opensnow_regions = ['arizona', 'british-columbia', 'california',
                            'colorado', 'lake-tahoe', 'new-mexico', 'oregon',
                            'utah', 'vermont', 'washington']
        base_region_url_template = Template(
            'https://www.onthesnow.com/${region}/skireport.html')
        region_urls = [base_region_url_template.substitute(region=region)
                       for region in opensnow_regions]
        for url in region_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        station_xpath = '//*[@id="resort-list-wrapper"]/div/table/tbody/tr'
        ## //*[@id="resort-list-wrapper"]/div/table/tbody/tr[3]/td/div/div[1]/a
        inspect_response(response, self)
        for item in response.xpath(station_xpath):
            item_url = item.xpath("div[@class='name link-light']").extract()
            inspect_response(response, self)            
            yield scrapy.Request(item_url, callback=self.parse_station)

    def parse_station(self, response):
        snowfall_xp = '*[@id="left_rail"]/div[1]/div[1]/div[2]/div/div[5]/script[1]/text()'
        yield {'snowdata': response.xpath(snowfall_xp).extract(),
               'station': response.request.url}

process.crawl(OpenSnowSpider)
process.start()