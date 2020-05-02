#!/bin/bash
# to set up ec2 instance to run scrapy-shell
yum -y install git
yum -y install python3
pip3 -y install virtualenv
python3 -m venv ./.venv
source ./.venv/bin/activate
#pip3 -y install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter 
pip3 -y install scrapy-splash
yum -y install docker
service docker stop && service docker start
docker run -p 8050:8050 scrapinghub/splash