# syntax=docker/dockerfile:1

FROM nvidia/cuda:10.2-base AS base
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl \
    unzip \
    python3 \
    python3-pip \
    python3-venv \
    dos2unix \
	ffmpeg libsm6 libxext6    

FROM base AS dependencies
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip 
COPY src/server/requirements.txt ./src/server/requirements.txt

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r src/server/requirements.txt 
RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt
RUN pip3 install torch torchvision
COPY . .

RUN pip3 install --upgrade setuptools
#CMD ls src/yolov3_helper -a
RUN cd src/yolov3_helper \
    dos2unix setup_yolo_framework.sh && dos2unix download_basketball_model.sh && bash setup_yolo_framework.sh && bash download_basketball_model.sh

FROM dependencies AS build
WORKDIR /app
RUN cd src/yolov3_helper/yolov3 \
    python3 detect.py --cfg cfg/basketball.cfg --weights ../basketball/weights/best.pt --source ../basketball/samples --names data/basketball.names --save-txt --classes 1 
COPY src/yolov3_helper/detect.py src/yolov3_helper/yolov3/detect.py
CMD ls -a src/yolov3_helper
EXPOSE 4000
CMD python3 src/server.py &
