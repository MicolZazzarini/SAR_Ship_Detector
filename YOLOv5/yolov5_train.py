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
load_dotenv()

# Parsing degli argomenti
parser = argparse.ArgumentParser(description='Train YOLOv5 model')
parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size')
parser.add_argument('-o', '--optimizer', type=str, required=True, choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type')
parser.add_argument('-w', '--workers', type=int, default=8, help='Number of workers for data loading')
args = parser.parse_args()

# Ottieni l'API key di WandB dal file .env
wandb_api_key = os.getenv('WANDB_API_KEY')

# Esegui il login a WandB se la chiave è disponibile
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    os.environ['WANDB_MODE'] = 'disabled'

print(f"Running YOLOv5")
model_name: str = "yolov5s"

model = YOLOv5(f"model/{model_name}.pt")

data = "datav5.yaml"
epochs = args.epochs
imgsz = 640
batch = args.batch_size
patience = 50
project = "output/"
train_name = f"{model_name}_epoch{epochs}_batch{batch}_train"
val_name = f"{model_name}_epoch{epochs}_batch{batch}_val"
test_name = f"{model_name}_epoch{epochs}_batch{batch}_test"
optimizer = args.optimizer
workers = args.workers
pretrained = f"model/{model_name}.pt"

# Start the emissions tracker
tracker = EmissionsTracker(log_level="WARNING")
try:
    tracker.start()
except Exception as e:
    print(f"Error starting emissions tracker: {e}")

YOLOv5.train(model, data="datav5.yaml", epochs=epochs, batch_size=batch)

model.train(data=data,
            epochs=epochs,
            batch_size=batch,
            imgsz=imgsz,
            optimizer=optimizer,
            patience=patience,
            project=project,
            name=train_name,
            workers=workers)

# Validate the model
model.val(data=data, project=project, name=val_name, workers=workers)

# Test the model
model.test(data=data, project=project, name=test_name, workers=workers)

# Define the output directory for the emissions data
output_directory = f"output/CO2/{train_name}"
os.makedirs(output_directory, exist_ok=True)

# Stop the emissions tracker and save the data
try:
    emissions_data = tracker.stop()
    output_file = os.path.join(output_directory, f"{train_name}_emissions.csv")
    emissions_df = pd.DataFrame([emissions_data])
    emissions_df.to_csv(output_file, index=False)
    print(f"Emissions data saved to {output_file}")
except Exception as e:
    print(f"Error stopping emissions tracker or saving data: {e}")


# Grad-CAM
def apply_gradcam(model, img_path, target_layer):
    model.eval()
    img = cv2.imread(img_path)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    def forward_hook(module, input, output):
        model.feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        model.gradients = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(img)
    class_idx = torch.argmax(output, dim=1).item()

    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    gradients = model.gradients[0]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    feature_maps = model.feature_maps[0]

    for i in range(feature_maps.shape[0]):
        feature_maps[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = cv2.resize(heatmap, (imgsz, imgsz))
    heatmap = heatmap / np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    handle_forward.remove()
    handle_backward.remove()

    return superimposed_img


try:
    # Example usage
    for img_file in os.listdir('/app/ultralytics/data_CV_all/images/val'):
        img_path = os.path.join('/app/ultralytics/data_CV_all/images/val', img_file)
        gradcam_img = apply_gradcam(model, img_path, model.model.model[-1])
        output_path = os.path.join('/app/ultralytics/output/gradcam/images', img_file)
        cv2.imwrite(output_path, gradcam_img)
        print(f"Grad-CAM saved to {output_path}")
except Exception as e:
    print(f"Error loading grad-CAM: {e}")
