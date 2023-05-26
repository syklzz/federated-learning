#!/bin/bash
sudo yum update -y

yes | sudo yum install git
sudo git clone https://github.com/syklzz/federated-learning.git /home/ec2-user/app

sudo yum install -y docker
sudo service docker start
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

yes | sudo yum install make