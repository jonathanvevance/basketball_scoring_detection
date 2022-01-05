#!/bin/bash

# Clone framework
curl -OL https://github.com/ultralytics/yolov3/archive/refs/tags/v7.zip && \
    unzip ./v7.zip && \
    mv ./yolov3-7 ./yolov3 && \
    rm ./v7.zip

# Getting pre-trained weights
cd ./yolov3
sh weights/download_yolov3_weights.sh

sudo apt-get update
sudo apt-get install unzip
sudo apt-get install python-dev
sudo apt-get install python-pip
sudo apt-get install python3-venv

# Setup virtual env
python3 -m venv .env
source .env/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install Cython
pip3 install numpy
pip3 install "pillow<7"
pip3 install -r requirements.txt
pip3 install torchvision   
pip3 install opencv-python # hack for error
deactivate

echo "*****************************************************************************"
echo "******                Your environment has been created                ******"
echo "*****************************************************************************"
echo ""
echo "If you had no errors, You can proceed to work with your virtualenv as normal."
echo "(run 'source .env/bin/activate' in your assignment directory to load the venv,"
echo " and run 'deactivate' to exit the venv. See assignment handout for details.)"