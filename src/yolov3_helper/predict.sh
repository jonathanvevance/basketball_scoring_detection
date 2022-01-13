cd yolov3

python3 detect.py \
  --cfg cfg/basketball.cfg \
  --weights ../basketball/weights/best.pt \
  --names data/basketball.names \
  --source $1 \
  --output $2 \
  --save-txt --classes 1 # filter for basket
