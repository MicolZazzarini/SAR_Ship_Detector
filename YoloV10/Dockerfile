FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

LABEL authors="Andrea Fiorani"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5dbus5 \
    qttools5-dev \
    qttools5-dev-tools \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libxfixes-dev \
    libx11-xcb-dev \
    libxcb-glx0-dev \
    git \
    unzip \
    wget \
    tmux \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-pip \
    sudo \
    zip \
    tmux

RUN wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz && \
    tar xzf Python-3.9.18.tgz && \
    cd Python-3.9.18 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.9 /usr/bin/python3.9 && \
    ln -s /usr/local/bin/pip3.9 /usr/bin/pip3.9 && \
    cd .. && \
    rm -rf Python-3.9.18.tgz Python-3.9.18

RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3
RUN rm /usr/bin/pip3 && ln -s /usr/local/bin/pip3.9 /usr/bin/pip3
RUN ln -s /usr/bin/python3.9 /usr/bin/python  # Link simbolico per python
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install setuptools==69.5.1 packaging==21.0
RUN python3.9 -m pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3.9 -m pip install torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN apt-get install -y g++-7 && apt-get install -y libstdc++6

RUN python3.9 -m pip install wandb python-dotenv codecarbon scikit-learn matplotlib opencv-python-headless ttach

WORKDIR /app

COPY requirements.txt .
RUN python3.9 -m pip install -r requirements.txt

COPY settings.yaml /root/.config/yolov5/settings.yaml

WORKDIR /app/ultralytics/

COPY datav10.yaml /app/ultralytics/
COPY YOLOv10sCustom.yaml /app/ultralytics/

VOLUME /app/output

RUN git clone https://github.com/THU-MIG/yolov10.git && \
    cd yolov10 && \
    python3.9 -m pip install -r requirements.txt && \
    python3.9 -m pip install -e .

COPY yolov10_tuning.py /app/ultralytics/

COPY yolo_cam /app/ultralytics/yolo_cam/
