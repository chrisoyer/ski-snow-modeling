#!/bin/bash

# script description: log into ec2, do initial config, and get into docker image

# exit on error, show steps
set -eoxu pipefail

# get active ec2 ipa
ec2ip=$(aws ec2 describe-instances | grep -oP '(?<=PublicDnsName": ")ec2\-.+\.com(?=",)' | head -1)

# login
# SECURITY note: for prod environments, strict host key check should be on. Only turn off for 
# connection because this is always the first time ec2 instance is conected to.
ssh -i "~/.aws/nvirgina-2.pem" -o StrictHostKeyChecking=no -L 8888:localhost:8888 ec2-user@$ec2ip -yv