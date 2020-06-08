# ski-snow-modeling
Exploring the relationship between ski resort snowfall, base depth, and other features. 

### Description of Goals
EDA to see how ski resort snowfall, season length, base depth, etc. vary by region. Modeling: using altitude, location, etc to predict season length, and time series analysis of base depth evolution.
Tools: Scrapy, seaborn, altair (vega-lite visualizations), Numpy/Pandas etc., time series analysis with statsmodels (building to regression with SARIMA errors), tensorflow LSTM, FBProphet. Because prophet is somewhat difficult to install (dependencies on Stan and C++ compiler), I created a docker image of the install and am running it on AWS. 

### From EDA including geographic data:
Data was scraped from wkikipedia (for geographic data including elevation) and OntheSnow.com (for time series data of snow fall and base depth by day.) Pandas and Scrapy were used for scraping. Data cleaning was made difficult because the data was non-randomly missing: the base and snowfall data is only reported when resorts are open. I assumed zero snowfall for unreported dates, assumed 

Elevation and Skiable Area, show lowest to highest elevation within each resort: ![link](./resources/elevation_vs_area.png "Still working on embedding actual vega visualization, not just saved image. This would show the interactivity of the graphic")

Link to large plot of ski resorts, grouped by region, showing elevation and annual snowfall: [link](./resources/elevation_by_region.png "This also is lacking the interactivity native vega visualization in full HTML woudl afford")

Example resort, A Basin. The base depth can be seen to be highly seasonal, with frequent jumps interspersed in an overall decrease (absent said jumps). Jumps should be powder days, and this is what I attempt to model. Note: 'pseudo_ts' is timestamp altered so all series within a regon are contiguous, to facilitate grouped analysis: ![link](./resources/Abasin_viz.png)

### Data cleaned for time series analysis and ARIMA style models explored. 



