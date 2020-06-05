FROM ubuntu:16.04

WORKDIR /app

COPY . /app
RUN apt-get -y update  && apt-get install -y \
  python3 \
  python3-dev \
  python3-pip \
  git \
  software-properties-common \
  apt-utils \
  python-dev \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools
RUN pip install cython
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install pystan
RUN pip install fbprophet
RUN pip install notebook
RUN pip install git

EXPOSE 8888

# Following CMD keeps the container running
# Modify CMD to run the app that you require. 
CMD tail -f /dev/null &
