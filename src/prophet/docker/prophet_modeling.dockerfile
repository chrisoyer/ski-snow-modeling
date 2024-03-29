#########
# Notes #
#########

# modified from scipy-notebook/Dockerfile
# pystan is hardlinked to version for ec2 RedHat;
# may need updated if source python version changes

# needs to be fixed to be multi-stage
ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER
WORKDIR /app

USER root

# Create the environment by adding layers:
# install compilers and pip
RUN apt-get update && \
    apt-get install --no-install-recommends gcc g++ python3-pip -y && \
    rm -rf /var/lib/apt/lists/*  # cleanup
	
COPY ./src/prophet/docker/requirements1.txt .
COPY ./src/prophet/docker/requirements2.txt .
COPY ./src/prophet/docker/requirements3.txt .
RUN pip install --no-cache-dir -r requirements1.txt # Cython etc.
RUN pip install --no-cache-dir -r requirements2.txt # pystan etc.
Run pip install --no-cache-dir -r requirements3.txt # prophet

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# CMD runs the default
CMD jupyter lab --port=8888 --ip=* --no-browser --allow-root --notebook-dir="/"

# copy repo into image - assumes dockerfile is in relevant repo
COPY . /app

# Doesn't directly expose port except for inter-container comm
EXPOSE 8888

# causes problems because jovyan user password is a secret!
# USER $NB_UID

