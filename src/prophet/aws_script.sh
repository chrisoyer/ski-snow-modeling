#!/bin/bash

# script description: log into ec2, do initial config, and get into docker image

# exit on error
set -e

# get active ec2 ipa
ec2ip=$(aws ec2 describe-instances | grep -oP '(?<=PublicDnsName": ")ec2\-.+\.com(?=",)' | head -1)

# login
# SECURITY note: for prod environments, strict host key check should be on. Only turn off for 
# connection because this is always the first time ec2 instance is conected to.
ssh -i "~/.aws/nvirgina-2.pem" -o StrictHostKeyChecking=no -L 8888:localhost:8888 ec2-user@$ec2ip -yv "

# update ec2
sudo yum update -y

# to use docker:
sudo amazon-linux-extras install docker

# Add the ec2-user to the docker group so you can execute Docker commands without using sudo:
sudo usermod -aG docker ec2-user && sudo groupadd docker && sudo newgrp docker && dockerd
service docker start

# git 
sudo yum install git -y
git clone https://github.com/chrisoyer/ski-snow-modeling.git

# build docker
cd ./ski-snow-modeling/src/prophet/docker/
docker build . -f prophet_modeling.dockerfile -t prophet_img:v1

# start docker container
docker run -it prophet_img

# close ssh string
"
