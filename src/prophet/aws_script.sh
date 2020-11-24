#!/bin/bash

# script description: log into ec2, do initial config, and get into docker image

# exit on error, show steps
set -eoxu pipefail

# launch instance
#aws ec2 run-instances --image-id ami-0947d2ba12ee1ff75 --instance-type t2.micro --security-group-ids jupyter-sg --iam Arn=arn:aws:iam::622780367867:instance-profile/data_science_role --key-name nvirgina-2

# make sure instance has launched
#sleep 45

# get active ec2 ipa
ec2ip=$(aws ec2 describe-instances | grep -oP '(?<=PublicDnsName": ")ec2\-.+\.com(?=",)' | head -1)

# login
# SECURITY note: for prod environments, strict host key check should be on. Only turn off for 
# connection because this is always the first time ec2 instance is conected to.
ssh -i "~/.aws/nvirgina-2.pem" -o StrictHostKeyChecking=no -L 8888:localhost:8888 ec2-user@$ec2ip -yv . ./ec2_script.sh
