import argparse
import os
import time
from dotenv import load_dotenv
from ultralytics import YOLO
import wandb
import pandas as pd
from codecarbon import EmissionsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 and run inference on images")
    parser.add_argument('-y', '--yaml_conf', type=str, default='yolov8s.pt', help='file conf for model yolov8')
    parser.add_argument('-n', '--project_name', type=str, default="output", help='Project name')
    parser.add_argument('-e', '--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-o', '--optimizer', type=str, default='SGD', choices=['AdamW', 'Adam', 'SGD'], help='Optimizer for training')
    parser.add_argument('-lr0', '--learning_rate_initial', type=float, default=0.001, help='Initial learning rate for training')
    parser.add_argument('-lrf', '--learning_rate_final', type=float, default=0.0001, help='Final learning rate for training')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005, help='Weight decay for optimizer')
    parser.add_argument('-wu', '--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        os.environ['WANDB_MODE'] = 'disabled'

    epochs = args.epochs
    batch_size = args.batch_size
    optimizer = args.optimizer
    learning_rate_initial = args.learning_rate_initial
    learning_rate_final = args.learning_rate_final
    momentum = args.momentum
    weight_decay = args.weight_decay
    warmup_epochs = args.warmup_epochs

    model_name = args.yaml_conf
    project = args.project_name
    train_name = f"{model_name}_epoch{epochs}_batch{batch_size}_op{optimizer}_lr0{learning_rate_initial}_lrf{learning_rate_final}_mom{momentum}_wd{weight_decay}_wu{warmup_epochs}_train"
    val_name = f"{model_name}_epoch{epochs}_batch{batch_size}_op{optimizer}_lr0{learning_rate_initial}_lrf{learning_rate_final}_mom{momentum}_wd{weight_decay}_wu{warmup_epochs}_val"

    dataset = "data.yaml"

    model = YOLO(args.yaml_conf)

    tracker = EmissionsTracker(log_level="WARNING")
    try:
        tracker.start()
    except Exception as e:
        print(f"Error starting emissions tracker: {e}")

    model.train(
        task="detect",
        data=dataset,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        optimizer=optimizer,
        project=project,
        name=train_name,
        lr0=learning_rate_initial,
        lrf=learning_rate_final,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        save=True,
        save_period=10,
        device=0
    )

    model.val(
        data=dataset,
        project=project,
        name=val_name
    )

    try:
        emissions_data = tracker.stop()
        output_directory = f"output/CO2/{train_name}"
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, f"{train_name}_emissions.csv")
        emissions_df = pd.DataFrame([emissions_data])
        emissions_df.to_csv(output_file, index=False)
        print(f"Emissions data saved to {output_file}")
    except Exception as e:
        print(f"Error stopping emissions tracker or saving data: {e}")


if __name__ == "__main__":
    main()
