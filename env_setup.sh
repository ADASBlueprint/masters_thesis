#!/bin/bash
# Update packages
apt-get update
# Non-interactive mode, use default answers
export DEBIAN_FRONTEND=noninteractive
# Workaround for libc6 bug - asking about service restart in non-interactive mode
# https://bugs.launchpad.net/ubuntu/+source/eglibc/+bug/935681
echo 'libc6 libraries/restart-without-asking boolean true' | debconf-set-selections
# Install Python 3.7
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y install python3.7 python3.7-dev
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.7
#install SUMO
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc -y
# # Add Nvidia repositories
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
# dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
# dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
# apt-get update
# # Install drivers, CUDA and cuDNN
# apt-get -y install --no-install-recommends nvidia-driver-440
# sudo apt-get install cuda-drivers-440 -y
# sudo apt-get install cuda-runtime-10-0 -y
# apt-get -y install --no-install-recommends cuda-10-0 libcudnn7=\*+cuda10.0 libcudnn7-dev=\*+cuda10.0
# #apt-get -y install --no-install-recommends cuda-10-1 libcudnn7=\*+cuda10.1 libcudnn7-dev=\*+cuda10.1
# apt-get -y install --no-install-recommends libnvinfer5=5.\*+cuda10.0 libnvinfer-dev=5.\*+cuda10.0
# #apt-get -y install --no-install-recommends libnvinfer5=5.\*+cuda10.1 libnvinfer-dev=5.\*+cuda10.1
# # Install TensorFlow
# pip3.7 install tensorflow-gpu==1.*
# # Install PyTorch
# pip3.7 install torch torchvision
# Install other Python packages
pip3.7 install numpy
pip3.7 install pandas
pip3.7 install matplotlib
pip3.7 install tqdm
pip3.7 install traci
pip3.7 install gym
pip3.7 install keras==2.3.1
# Reboot
# reboot