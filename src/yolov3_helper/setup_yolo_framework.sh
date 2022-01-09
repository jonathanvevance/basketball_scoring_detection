#!/bin/bash

# Clone framework
curl -OL https://github.com/ultralytics/yolov3/archive/refs/tags/v7.zip && \
    unzip ./v7.zip && \
    mv ./yolov3-7 ./yolov3 && \
    rm ./v7.zip

# Getting pre-trained weights
cd ./yolov3
sh weights/download_yolov3_weights.sh

echo "*****************************************************************************"
echo "******                Your environment has been created                ******"
echo "*****************************************************************************"
echo ""