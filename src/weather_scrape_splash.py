#!/usr/bin/env python
# coding: utf-8

import scrapy
from scrapy_splash import SplashRequest
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.shell import inspect_response
from string import Template

process = CrawlerProcess({'AUTOTHROTTLE_ENABLED': True,  # or download delay
                          'HTTPCACHE_ENABLED': True,  # remove for final scrape to get live data
                          'ROBOTSTXT_OBEY': True,
                          'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36',
                          'FEED_URI': 'ski_weather_data.json',
                          'FEED_FORMAT': 'json',
                          'FEED_EXPORT_ENCODING': '"utf-8"'})

class OpenSnowSpider(scrapy.Spider):
    name = 'opensnow'
    origin_urls = [r'https://www.onthesnow.com/united-states/skireport.html']

    def start_requests(self):
        for url in self.origin_urls:
            yield SplashRequest(
                url=url, callback=self.parse_start, endpoint='render.html'
                                )    
    
    def parse_start(self, response):
        station_xpath = '//*[@id="resort-list-wrapper"]/div/table/tbody/tr'
        for item in response.xpath(station_xpath):
            print(f'found item {item}')
            subitems = item.xpath(station_xpath)
            for subitem in subitems:
                subitem_url = subitem.xpath('//td/div/div[1]/a/text()').extract()
                print(subitem_url)
                yield scrapy.Request(subitem_url, callback=self.parse_station)

    def parse_station(self, response):
        snowfall_xp = '*[@id="left_rail"]/div[1]/div[1]/div[2]/div/div[5]/script[1]/text()'
        yield {'snowdata': response.xpath(snowfall_xp).extract(),
               'station': response.request.url}
##########
# example
##########

#class MySpider(scrapy.Spider):
#    name = "jsscraper"
#    start_urls = ["http://quotes.toscrape.com/js/"]

#    def start_requests(self):
#        for url in self.start_urls:
#        yield SplashRequest(
#            url=url, callback=self.parse, endpoint='render.html'
#        )

#    def parse(self, response):
#        for q in response.css("div.quote"):
#        quote = QuoteItem()
#        quote["author"] = q.css(".author::text").extract_first()
#        quote["quote"] = q.css(".text::text").extract_first()
#        yield quote
        
########################
process.crawl(OpenSnowSpider)
process.start()