# README

## Table of Contents
1. [Introduction](#introduction)
2. [YOLO Versions](#yolo-versions)
   - [YOLOv8](#yolov8)
   - [YOLOv10](#yolov10)
3. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)
4. [Datasets](#datasets)
   - [HRSID](#hrsid)
   - [CAESAR Openship 2.0 SAR](#caesar-openship-20-sar)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction

This repository provides an overview of three versions of the YOLO (You Only Look Once) object detection framework: YOLOv5, YOLOv8, and YOLOv10. Additionally, it includes information about two key datasets used for training and evaluation: HRSID and CAESAR Openship 2.0 SAR.

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

## Getting Started

### Quickstart
To get started with any of the YOLO versions, you need to install the necessary dependencies. The following instructions will guide you through the installation process.


### Usage
After installing the dependencies, you can start using YOLOv5, YOLOv8, or YOLOv10 for object detection tasks. Below are basic usage examples for each version.


## YOLOv8s
### Traning
GPU type: NVIDIA RTX A6000  
CPU type: Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz
Task: Training detection ship from SAR

| CO2 Kg                | Epoche | Batch Size | Optimizer |
|-----------------------|--------|------------|-----------|
| 1.928400235849826     | 200    | 32         | SGD       |
| 1.8210006887608925    | 200    | 64         | SGD       |
| 1.726629182765182     | 200    | 128        | AdamW     |
| 1.72663               | 200    | 128        | SGD       |
| 2.706262813720201     | 300    | 64         | SGD       |
| **Totale**            | -      | -          | -         |
| **9.908922921096101** | -      | -          | -         |

### Inference img nave_3_rumore.jpg
CPU type: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz

| time in s | Epoche | Batch Size | Optimizer |
|-----------|--------|------------|-----------|
| 3.2085    | 200    | 32         | SGD       |
| 3.1526    | 200    | 64         | SGD       |
| 3.2191    | 200    | 128        | AdamW     |
| 3.1579    | 200    | 128        | SGD       |
| 3.1625    | 300    | 64         | SGD       |


## YOLOv10s
### Traning
GPU type: NVIDIA RTX A6000  
CPU type: Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz
Task: Training detection ship from SAR

| CO2 Kg                 | Epoche | Batch Size | Optimizer |
|------------------------|--------|------------|-----------|
| 2.3467992712042625     | 200    | 32         | SGD       |
| 2.225889854361914      | 200    | 64         | SGD       |
| 2.218272748326967      | 200    | 128        | AdamW     |
| 2.2214224895266463     | 200    | 128        | SGD       |
| 3.419242134353886      | 300    | 64         | SGD       |
| **Totale**             | -      | -          | -         |
| **12.431626497673677** | -      | -          | -         |


### Inference img nave_3_rumore.jpg
CPU type: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz

| time in s | Epoche | Batch Size | Optimizer |
|-----------|--------|------------|-----------|
|           | 200    | 32         | SGD       |
|           | 200    | 64         | SGD       |
|           | 200    | 128        | AdamW     |
|           | 200    | 128        | SGD       |
|           | 300    | 64         | SGD       |


