# ski-snow-modeling
Exploring the relationship between ski resort snowfall, base depth, and other features. 

### Description of Goals
EDA to see how ski resort snowfall, season length, base depth, etc. vary by region. Modeling: using altitude, location, etc to predict season length, and time series analysis of base depth evolution.
Tools: Scrapy, seaborn, altair (vega-lite visualizations), Numpy/Pandas etc., time series analysis with statsmodels (building to regression with SARIMA errors), tensorflow LSTM, FBProphet. Because prophet is somewhat difficult to install (dependencies on Stan and C++ compiler), I created a docker image of the install and am running it on AWS. 

### From EDA including geographic data:
Data was scraped from wkikipedia (for geographic data including elevation) and OntheSnow.com (for time series data of snow fall and base depth by day.) Pandas and Scrapy were used for scraping. 

Elevation and Skiable Area, show lowest to highest elevation within each resort: ![link](./resources/elevation_vs_area.png "Still working on embedding actual vega visualization, not just saved image. This would show the interactivity of the graphic")

Link to large plot of ski resorts, grouped by region, showing elevation and annual snowfall: [link](./resources/elevation_by_region.png "This also is lacking the interactivity native vega visualization in full HTML woudl afford")

Example resort, A Basin. The base depth can be seen to be highly seasonal, with frequent jumps interspersed in an overall decrease (absent said jumps). Jumps should be powder days, and this is what I attempt to model. Note: 'pseudo_ts' is timestamp altered so all series within a regon are contiguous, to facilitate grouped analysis: ![link](./resources/Abasin_viz.png)

Season length varied by region, with significant overlap ![link](./resources/season_length.png)

### Data Cleaning
I windsorized values on the right tail at 2.5 standard deviations: there were some values in the original data where decimals were missing (eg 65 inches followed the next day by 655 inches of base); there were replaced by prior good value. 
![link](./resources/log_base.png)

Data cleaning issues included data non-randomly missing: the base and snowfall data is only reported when resorts are open. I assumed zero snowfall for unreported dates, assumed all dates in August had 0 base depth (which should hold for all but one or two locations in the US with glacier skiing), and used 2nd order polynomial interpolation in between season end and the summer zero values. ![link](./resources/interpolated_data.png TODO: indicate filled values vs original)

Final average daily snowfall in each region by month: ![link](./resources/daily_snowfall.png)

TODO: Regress features against season length, e.g. snowfall, region, altitude, etc. 

### Data cleaned for time series analysis and ARIMA style models explored. 

Typical decomposition plot, this one for Winter Park: ![link](./resources/WP_decomposition.png)

Select model (choosing (p,d,q)(P,D,Q)s order terms): Working on setup of walk-forward crossvalidation of models. AIC/BIC based model selection works, but givin different top models compared to traditional selection of terms based on AC/PAC plots. Plots looke like this: ![link](./resources/AC_PAC.png). 

Models still in progress: LSTM model in Tenorflow.
Facebook Prophet GAM model. (dockfile is working, modeling still in progress).

