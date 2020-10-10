#########
# Notes #
#########

# modified from scipy-notebook/Dockerfile

# needs to be fixed to be multi-stage
# needs mid size ec2 instance; micro will not have enough RAM to compile
ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER
WORKDIR /app

USER root

# Create the environment by adding layers:
# install compilers and pip
RUN apt-get update && \
    apt-get install --no-install-recommends gcc g++ python3-pip -y && \
    rm -rf /var/lib/apt/lists/*  # cleanup
	
RUN sudo python -m pip install --upgrade pip

COPY requirements1.txt .
COPY requirements2.txt .
RUN pip install --no-cache-dir -r requirements1.txt # needs to be installed first
RUN pip install --no-cache-dir -r requirements2.txt

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# CMD runs the default
CMD ["jupyter", "notebook", "--port=8888", "--ip=*", "--no-browser", "--allow-root", '--notebook-dir="/"']

# Doesn't directly expose port except for inter-container comm
EXPOSE 8888

USER $NB_UID

