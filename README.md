# README

## Table of Contents
1. [Introduction](#introduction)
2. [YOLO Versions Used](#yolo-versions)
3. [Datasets](#datasets)
4. [Getting Started with Docker ( Training )](#getting-started)
5. [Getting Started with Notebook ( Inference )](#Notebook)
6. [Contributing](#contributing)
7. [License](#license)

---
## Introduction

This repository provides an overview of three versions of the YOLO (You Only Look Once) object detection framework: YOLOv8 and YOLOv10. Additionally, it includes information about two key datasets used for training and evaluation: HRSID and CAESAR Openship 2.0 SAR.

---
## YOLO Versions

### YOLOv8

YOLOv8 builds upon the advancements of YOLOv5 with further optimizations and new features aimed at improving performance in various object detection scenarios. It includes architectural changes and better handling of smaller objects.

**Key Features:**
- Enhanced model architecture
- Better performance on smaller objects
- Advanced augmentation techniques
- Further speed improvements

### YOLOv10

YOLOv10 represents the latest iteration in the YOLO series, featuring cutting-edge improvements in both model architecture and training techniques. This version is designed to provide state-of-the-art performance across a wide range of object detection tasks.

**Key Features:**
- State-of-the-art detection performance
- Highly optimized for various hardware
- Advanced training methodologies
- Comprehensive support for diverse datasets

---



## Datasets

---
## Getting Started with Docker

The procedure described is for YOLOv8. The only difference for YOLOv10 is to start in the YOLOv10 folder.

### 0. Configuration

Before starting the process, you need to configure the `.env` file by adding the `WANDB_API_KEY`. This key is necessary for logging and tracking your machine learning experiments with Weights & Biases (WandB).

The `.env` file should look like this:

```dotenv
WANDB_API_KEY=<api_key>
```

### 1. Generate Docker image and start the container
Start by building the Docker image and running the container for YOLOv8.

```bash
# Build Docker image for YOLOv8
docker build -t yolov8_image .

# Run the container with GPU support for YOLOv8
docker run -it --gpus '"device=2"' --name yolov8-container  --env-file .env -v $(pwd)/output:/app/ultralytics/output yolov8_image /bin/sh
```

This will create and start the container `yolov8-container`.

### 2. Connect to the running container
If the container is already running but you need to reconnect to it:

```bash
# Verify the container is running
docker ps

# Attach to the YOLOv8 container's shell
docker attach yolov8-container-ID
```

### 3. Create a Tmux session to manage long-running processes
To start managing long-running processes inside the container, create a Tmux session:

```bash
# Create a new tmux session called 'training'
tmux new-session -s training
```

Inside the Tmux session, execute your training script:

```bash
# Run the YOLOv8 training process
python3.9 yolov8_train_val_test.py -e 200 -b 32 -o SGD
```

### 4. Detach from Tmux to keep the process running in the background
After starting the training, detach from the Tmux session to leave it running in the background:

```bash
# Detach from the tmux session
Ctrl + B, then D
```

### 5. Exit the container without stopping it
To exit the Docker container without stopping it, use the following command:

```bash
# Detach from the container
Ctrl + P, then Ctrl + Q
```

This will return you to your local terminal while keeping the container and processes running.

### 6. Reattach to the Tmux session for monitoring or modifications
If you need to check on the training process or make modifications:

```bash
# Reattach to the YOLOv8 container
docker attach yolov8-container-ID

# Reattach to the tmux session
tmux attach-session -t training
```

### 7. Exit Tmux and stop the container
Once your work is complete, exit the Tmux session and stop the Docker container:

```bash
# Exit the tmux session
Ctrl + B, then exit

# Stop the container from a separate terminal
docker stop yolov8-container-ID
```

---
## Getting Started with Notebook

---
## Contributing

This project is developed and maintained by the following authors:

- 👩‍💻 [**Zazzarini Micol**](https://github.com/MicolZazzarini)
- 👨‍💻 [**Fiorani Andrea**](https://github.com/125ade)

---
## License
This project is licensed under the [MIT License](LICENSE) - Refer to the [LICENSE](LICENSE) file for more details.