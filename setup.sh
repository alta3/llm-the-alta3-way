#!/bin/bash
sudo apt update 
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
sudo -E apt-get install -y python3-pip python3-venv git
python3 -m pip install --upgrade --user pip
python3 -m pip install --user ansible
