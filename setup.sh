# Setup virtual env
sudo apt-get update
sudo apt-get install unzip
sudo apt-get install python3-dev
sudo apt-get install python3-pip
sudo apt-get install python3-venv

python3 -m venv .env
source .env/bin/activate
pip3 install --upgrade pip
pip3 install numpy
pip3 install opencv-python
pip3 install torch
pip3 install torchvision
pip3 install click
pip3 install tqdm
pip3 install scikit-learn
pip3 install matplotlib
deactivate
