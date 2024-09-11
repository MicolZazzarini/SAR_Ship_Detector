# README

## Table of Contents
1. [Introduction](#introduction)
2. [YOLO Versions Used](#yolo-versions)
3. [Datasets](#datasets)
4. [Getting Started with Docker ( Training )](#Getting-Started-with-Docker)
5. [Getting Started with Notebook ( Inference )](#Getting-Started-with-Notebook)
6. [YOLOvs8 Training and Inference Optimization Notebook](#YOLOv8-Training-and-Inference-Optimization-Notebook)
7. [Contributing](#contributing)
8. [License](#license)

---
## Introduction

This repository is designed to provide an overview and guidance on using two versions of the YOLO (You Only Look Once) object detection framework: YOLOv8 and YOLOv10. YOLO is a state-of-the-art deep learning model for object detection, offering real-time detection capabilities across a variety of tasks. YOLOv8 builds on the innovations introduced by YOLOv5, incorporating improved model architecture and optimized detection for smaller objects. YOLOv10 represents the latest advancement in the YOLO series, featuring cutting-edge performance with enhanced training methodologies and support for diverse datasets.

In this project, two key datasets were used: the HRSID dataset, consisting of high-resolution satellite images for ship detection, and the CAESAR Openship 2.0 SAR dataset, which provides a large-scale collection of SAR images. The HRSID dataset was used during the exploratory phase, offering insights into initial detection tasks, while the CAESAR Openship 2.0 SAR dataset was employed for the main training phase to further refine and improve the model's performance.

The repository also includes instructions for setting up the environment, using Docker for training, and utilizing notebooks for inference. Contributions and code modifications are encouraged, and the project is licensed under the MIT License.

---
## YOLO Versions

### YOLOv8

YOLOv8 builds upon the advancements of YOLOv5 with further optimizations and new features aimed at improving performance in various object detection scenarios. It includes architectural changes and better handling of smaller objects.

**Key Features:**
- Enhanced model architecture
- Better performance on smaller objects
- Advanced augmentation techniques
- Further speed improvements

The code for YOLOv8 was taken from the reference repository [Ultralytics YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics).


### YOLOv10

YOLOv10 represents the latest iteration in the YOLO series, featuring cutting-edge improvements in both model architecture and training techniques. This version is designed to provide state-of-the-art performance across a wide range of object detection tasks.

**Key Features:**
- State-of-the-art detection performance
- Highly optimized for various hardware
- Advanced training methodologies
- Comprehensive support for diverse datasets

The code for YOLOv10 was taken from the reference repository [YOLOv10 GitHub Repository](https://github.com/THU-MIG/yolov10).


---
## Datasets

### HRSID Dataset
The HRSID (High-Resolution Ship Detection) dataset is used for ship detection tasks, primarily in optical remote sensing images. It consists of 5600 high-resolution satellite images containing ships of various sizes and orientations. This dataset is ideal for an initial exploratory analysis, providing a broad range of ship images in different maritime environments. We used the HRSID dataset during the exploratory phase of our project to gain insights into the ship detection task before moving to more complex datasets.

Repository link: [HRSID GitHub Repository](https://github.com/chaozhong2010/HRSID).

### CAESAR Openship 2.0 SAR Dataset
The CAESAR Openship 2.0 SAR (Synthetic Aperture Radar) dataset is a large-scale collection of SAR images designed for ship detection and classification. It contains over 80,000 labeled images of ships captured in various conditions, making it suitable for advanced object detection tasks. We used this dataset for the main training phase of our models, as its complexity and detail allowed us to improve accuracy significantly. The SAR data offers resilience to challenging weather conditions, enhancing real-world applicability in maritime ship detection.

Repository link: [CAESAR Openship 2.0 SAR GitHub Repository](https://github.com/CAESAR/Openship-2.0).

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

## YOLOv8 Training and Inference Optimization Notebook

This repository contains a Google Colab notebook demonstrating how to train a YOLOv8 model on a custom dataset, perform inference, and optimize the model using TensorRT for faster inference. The notebook follows these steps:
1. Mount Google Drive
2. Set up the YOLOv8 environment
3. Train YOLOv8 models (original and modified)
4. Perform inference with trained models
5. Optimize the YOLOv8 model using TensorRT
6. Perform inference with the TensorRT-optimized model

### Dataset Structure

The dataset should be organized in the following structure:
```
dataset/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
└── labels/
├── train/
├── val/
└── test/
 ```
### Overview

#### Training the Original YOLOv8 Model

The original YOLOv8 model architecture is used to train the model on the dataset. This model is designed for general object detection tasks.

#### Optimizing YOLOv8 for Small Objects

To optimize the YOLOv8 model for detecting small objects, certain layers in the architecture are removed. This modification helps in focusing the model's capacity on smaller scales of the input image, making it more efficient in detecting small objects.

##### Original YOLOv8 Architecture

![Original YOLOv8 Architecture](https://www.stunningvisionai.com/course/yolov8-architecture.png)

##### Modified YOLOv8 Architecture for Small Objects

The following diagram shows the modifications made to the YOLOv8 architecture to better handle small object detection:

![Modified YOLOv8 Architecture for Small Objects](https://www.stunningvisionai.com/course/yolov8-architecture-modification-for-small-object.png)

##### Final YOLOv8 Architecture for Small Objects

![Final YOLOv8 Architecture for Small Objects](https://www.stunningvisionai.com/course/yolov8-architecture-for-small-object.png)

#### YOLOv8 Configuration for Small Objects

The following is the YOLOv8 architecture configuration file modified for detecting small objects, where unnecessary layers are commented out:

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
#scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#  # [depth, width, max_channels]
#  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

depth_multiple: 0.33
width_multiple: 0.50
max_channels: 1024

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
#  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9 -> 7

# YOLOv8.0n head
head:
#  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small) -> 10

#  - [-1, 1, Conv, [256, 3, 2]]
#  - [[-1, 12], 1, Concat, [1]] # cat head P4
#  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

#  - [-1, 1, Conv, [512, 3, 2]]
#  - [[-1, 9], 1, Concat, [1]] # cat head P5
#  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[10], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

### TensorRT Optimization

#### What is TensorRT?

TensorRT is a deep learning inference optimizer and runtime library developed by NVIDIA. It is designed to deliver high-performance deep learning inference, leveraging NVIDIA GPUs. TensorRT takes a trained model and optimizes it for inference by performing layer fusion, precision calibration, kernel auto-tuning, and other optimizations to accelerate the inference process.

#### Benefits of TensorRT Optimization

- **Increased Throughput**: By optimizing the model, TensorRT can significantly increase the number of inferences per second.
- **Reduced Latency**: Optimizations reduce the time taken for each inference, making real-time applications more responsive.
- **Lower Power Consumption**: Efficient use of GPU resources can lead to reduced power consumption, which is crucial for edge devices.

#### How to Optimize YOLOv8 with TensorRT

In the notebook, we demonstrate how to convert a trained YOLOv8 model to a TensorRT engine. The process involves:

1. Installing necessary libraries: `tensorrt`, `onnx`, `onnxsim`, and `onnxruntime-gpu`.
2. Exporting the YOLOv8 model to ONNX format.
3. Converting the ONNX model to a TensorRT engine.

The following code snippets in the notebook perform these steps:

```python
# Install required libraries
!pip install tensorrt tensorrt_lean tensorrt_dispatch
!pip install onnx onnxsim onnxruntime-gpu

# Static variable for TensorRT export
EXPORT_NAME = 'runs/detect/yolov8_ships_small/weights/best.pt'

# Export YOLOv8 Model to TensorRT
!yolo export model={EXPORT_NAME} format=engine half=True device=0
```
### Running the Notebook

1. Clone this repository or download the notebook.
2. Open the notebook in Google Colab.
3. Follow the steps in the notebook to mount your Google Drive, set up the YOLOv8 environment, and train the models.
4. Perform inference and visualize the results.
5. Optimize the model using TensorRT and perform inference with the optimized model.

### Acknowledgments

This work leverages the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) repository and the related documentation. For additional insights and performance improvement techniques, you may also find the [YOLO Performance Improvement Masterclass on Udemy](https://www.udemy.com/course/yolo-performance-improvement-masterclass) useful.


---
## Contributing

This project is developed and maintained by the following authors:

- 👩‍💻 [**Zazzarini Micol**](https://github.com/MicolZazzarini)
- 👨‍💻 [**Fiorani Andrea**](https://github.com/125ade)

---
## License
This project is licensed under the [MIT License](LICENSE) - Refer to the [LICENSE](LICENSE) file for more details.
