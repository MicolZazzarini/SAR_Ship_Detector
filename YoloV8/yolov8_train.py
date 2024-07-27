from ultralytics import YOLO, __version__
from dotenv import load_dotenv
import os
import wandb
import pandas as pd
from codecarbon import EmissionsTracker
# in ultralytics installare requirements poi fare pip install -e .
# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni l'API key di WandB dal file .env
wandb_api_key = os.getenv('WANDB_API_KEY')

# Esegui il login a WandB se la chiave è disponibile
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    os.environ['WANDB_MODE'] = 'disabled'

print(f"Running YOLO {__version__}")
model_name = 'yolov8s'

model = YOLO(f"{model_name}.yaml")


data = "data.yaml"
epochs = 300
imgsz = 640
batch = 64
patience = 50
project = f"output/"
train_name = f"{model_name}_epoch{epochs}_batch{batch}_train"
val_name = f"{model_name}_epoch{epochs}_batch{batch}_val"
test_name = f"{model_name}_epoch{epochs}_batch{batch}_test"
optimizer = "SGD"
pretrained = f"model/{model_name}.pt"


# Start the emissions tracker
tracker = EmissionsTracker(log_level="WARNING")
try:
    tracker.start()
except Exception as e:
    print(f"Error starting emissions tracker: {e}")


model.train(data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            optimizer=optimizer,
            patience=patience,
            pretrained=pretrained,
            name=train_name)

# Validate the model
model.val(project=project, name=val_name)

# Test the model
model.val(data=data, split='test', project=project, name=test_name)
# Define the output directory for the emissions data
output_directory = f"output/CO2/{train_name}"
os.makedirs(output_directory, exist_ok=True)

# Stop the emissions tracker and save the data
try:
    emissions_data = tracker.stop()
    output_file = os.path.join(output_directory, f"{train_name}emissions.csv")
    emissions_df = pd.DataFrame([emissions_data])
    emissions_df.to_csv(output_file, index=False)
    print(f"Emissions data saved to {output_file}")
except Exception as e:
    print(f"Error stopping emissions tracker or saving data: {e}")
