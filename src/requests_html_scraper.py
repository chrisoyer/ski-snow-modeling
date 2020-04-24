#!/usr/bin/env python
# coding: utf-8

from requests_html import HTMLSession
import pyppdf.patch_pyppeteer

start_url = r'https://www.onthesnow.com/united-states/skireport.html'


session = HTMLSession()
r = session.get(start_url, verify=False) #'../data/root_ca.pem')
r.html.render(scrolldown=30, sleep=4)
link = [htm.absolute_links for htm in r.html]
links = ",".join(link)


with open("../data/station_links.txt", "w") as text_file:
    text_file.write(",".join(links))

