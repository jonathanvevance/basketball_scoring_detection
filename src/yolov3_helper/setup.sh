# setup yolov3
sudo apt-get update
sudo apt-get -y install unzip
sudo apt-get -y install dos2unix

dos2unix setup_yolo_framework.sh
dos2unix download_basketball_model.sh

sudo bash setup_yolo_framework.sh
sudo bash download_basketball_model.sh

# TEST 
cd yolov3
source .env/bin/activate

python3 detect.py \
  --cfg cfg/basketball.cfg \
  --weights ../basketball/weights/best.pt \
  --source ../basketball/samples \
  --names data/basketball.names \
  --save-txt --classes 1 # filter for basket
