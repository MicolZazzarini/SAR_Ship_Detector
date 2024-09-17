from yolov5 import YOLOv5
from dotenv import load_dotenv
import os
import wandb
import pandas as pd
from codecarbon import EmissionsTracker
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Carica le variabili d'ambiente dal file .env


# Parsing degli argomenti
parser = argparse.ArgumentParser(description='Train YOLOv5 model')
parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size')
parser.add_argument('-o', '--optimizer', type=str, required=True, choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type')
parser.add_argument('-w', '--workers', type=int, default=8, help='Number of workers for data loading')
args = parser.parse_args()


load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    os.environ['WANDB_MODE'] = 'disabled'

model_name: str = "yolov5s"

model = YOLOv5(f"model/{model_name}.pt")

data = "datav5.yaml"
epochs = args.epochs
imgsz = 640
batch = args.batch_size
patience = 50
project = "output/"
train_name = f"{model_name}_e{epochs}_b{batch}_train"
val_name = f"{model_name}_e{epochs}_b{batch}_val"
test_name = f"{model_name}_e{epochs}_b{batch}_test"
optimizer = args.optimizer
workers = args.workers
pretrained = f"model/{model_name}.pt"

tracker = EmissionsTracker(log_level="WARNING")
try:
    tracker.start()
except Exception as e:
    print(f"Error starting emissions tracker: {e}")

YOLOv5.train(model, data="datav5.yaml", epochs=epochs, batch_size=batch)

model.train(
    data=data,
    epochs=epochs,
    batch_size=batch,
    imgsz=imgsz,
    optimizer=optimizer,
    patience=patience,
    project=project,
    name=train_name,
    workers=workers
)

model.val(
    data=data,
    project=project,
    name=val_name,
    workers=workers
)

model.test(
    data=data,
    project=project,
    name=test_name,
    workers=workers
)

try:
    output_directory = f"output/CO2/{train_name}"
    os.makedirs(output_directory, exist_ok=True)
    emissions_data = tracker.stop()
    output_file = os.path.join(output_directory, f"{train_name}_emissions.csv")
    emissions_df = pd.DataFrame([emissions_data])
    emissions_df.to_csv(output_file, index=False)
    print(f"Emissions data saved to {output_file}")
except Exception as e:
    print(f"Error stopping emissions tracker or saving data: {e}")

