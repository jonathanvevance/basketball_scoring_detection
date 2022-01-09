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
RUN cd src/yolov3_helper \
    dos2unix setup_yolo_framework.sh && dos2unix download_basketball_model.sh && bash setup_yolo_framework.sh && bash download_basketball_model.sh

FROM dependencies AS build
WORKDIR /app
RUN cd src/yolov3_helper/yolov3 \
    python3 detect.py --cfg cfg/basketball.cfg --weights ../basketball/weights/best.pt --source ../basketball/samples --names data/basketball.names --save-txt --classes 1
COPY src/yolov3_helper/detect.py src/yolov3_helper/yolov3/detect.py

### From readme
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
# RUN nvm install --lts

### Solution 1 - https://stackoverflow.com/a/60137919
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh && \
#     /root/nvm_install.sh && \
#     source /root/.bashrc && \
#     cd /root && \
#     nvm install --lts

### Solution 2
# SHELL ["/bin/bash", "--login", "-i", "-c"]
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
# RUN source /root/.bashrc && nvm install --lts
# SHELL ["/bin/bash", "--login", "-c"]

### Solution 3
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash && \
#     . /root/.bashrc && \
#     cd /root && \
#     nvm install --lts

### Solution 4
# ENV NVM_DIR /usr/local/nvm
# RUN curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.39.0/install.sh | bash
# ENV NODE_VERSION v8.1.2
# RUN /bin/bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && nvm use --delete-prefix $NODE_VERSION"

# ENV NODE_PATH $NVM_DIR/versions/node/$NODE_VERSION/lib/node_modules
# ENV PATH      $NVM_DIR/versions/node/$NODE_VERSION/bin:$PATH

### Solution 5 - THIS WORKED
ENV NVM_DIR /root/.nvm
RUN curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.39.0/install.sh | bash
ENV NODE_VERSION v8.1.2
RUN /bin/bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && nvm use --delete-prefix $NODE_VERSION"

ENV NODE_PATH $NVM_DIR/versions/node/$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/$NODE_VERSION/bin:$PATH

# --------------------------
RUN node --version
RUN npm --version

RUN cd src/server \
    npm install && npm install -g serve

EXPOSE 4000

COPY docker_script.sh docker_script.sh
CMD ./docker_script.sh
