#!/bin/bash
# TODO: ansiblizet 
export DEBIAN_FRONTEND=noninteractive 
export NEEDRESTART_MODE=a

sudo apt update
sudo apt --purge remove -y "*cublas*" "cuda*" "*nvidia*"
sudo apt clean -y
sudo apt autoremove -y
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

sudo -E apt install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update

sudo -E apt install -y libnvidia-common-520
sudo -E apt install -y libnvidia-gl-520
sudo -E apt install -y nvidia-driver-520

sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update

sudo -E apt -y install cuda-toolkit-11-8
echo "reboot required..."

echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

sudo -E apt -y install python3 python3-venv python3-pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

nvidia-smi
nvcc -V
