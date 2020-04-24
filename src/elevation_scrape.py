#!/usr/bin/env python
# coding: utf-8

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from string import Template

process = CrawlerProcess({'AUTOTHROTTLE_ENABLED': True,  # or download delay
                          'HTTPCACHE_ENABLED': True,  # remove for final scrape to get live data
                          'ROBOTSTXT_OBEY': True,
                          'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36',
                          'FEED_URI': './data/ski_station_data.json',
                          'FEED_FORMAT': 'json',
                          'FEED_EXPORT_ENCODING': '"utf-8"'})


class WikiSkiStationSpider(scrapy.Spider):
    name = 'wikistation'
    start_urls = [r"https://en.wikipedia.org/wiki/List_of_ski_areas_and_resorts_in_the_United_States"]

    def parse(self, response):
        station_row_xpath = r'//*[@id="mw-content-text"]/div/ul'
        for item in response.xpath(station_row_xpath):
            for subitem in item.xpath("li"):
                item_url = item.xpath("/a[1]").extract()
                yield scrapy.Request(item_url, callback=self.parse_station)

    def parse_station(self, response):
        starting_xpath = r'//*[@id="mw-content-text"]/div/'
        target_fields = ["Top elevation", "Base elevation", "Vertical", 
                         "Coordinates", "Snowfall",]
        results = {}
        for tr_path in response.xpath(starting_xpath):
            for table_path in tr_path.xpath("table"):
                for tbody_path in table_path.xpath("tbody"):
                    for target in target_fields.xpath("tr"):
                        if tbody_path.xpath(f"th[text()={target}]"):
                            results[target] = tbody_path.xpath("td")
                
        yield {'station': response.request.url, "results": results}

process.crawl(WikiSkiStationSpider)
process.start()