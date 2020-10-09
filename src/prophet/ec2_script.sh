# update ec2
sudo yum update -y

# to use docker:
sudo amazon-linux-extras install docker

# Add the ec2-user to the docker group so you can execute Docker commands without using sudo:
#sudo usermod -aG docker ec2-user && sudo groupadd docker && sudo newgrp docker && dockerd
service docker start

# git
sudo yum install git -y
git clone https://github.com/chrisoyer/ski-snow-modeling.git

# build & start container. Use host networking to avoid ip forwarding -p 8888:8888
sudo docker build . -f prophet_modeling.dockerfile -t prophet_img:v1 --network=host
sudo docker run -it prophet_image:v1
