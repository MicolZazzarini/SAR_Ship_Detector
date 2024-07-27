import argparse
import os
from yolov10.ultralytics import YOLOv10
from dotenv import load_dotenv
import wandb
import pandas as pd
from codecarbon import EmissionsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv10")
    parser.add_argument('-n', '--project_name', type=str, default="output", help='project name')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', choices=['AdamW', 'Adam', 'SGD'], help='optimizer for training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Get the WandB API key from environment variables
    wandb_api_key = os.getenv('WANDB_API_KEY')

    # Login to WandB if the API key is available
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        os.environ['WANDB_MODE'] = 'disabled'

    epochs = args.epochs
    batch_size = args.batch_size
    optimizer = args.optimizer

    model_name = 'yolov10s'
    project = args.project_name
    train_name = f"{model_name}_epoch{args.epochs}_batch{args.batch_size}_op{optimizer}_train"
    val_name = f"{model_name}_epoch{args.epochs}_batch{args.batch_size}_op{optimizer}_val"
    test_name = f"{model_name}_epoch{args.epochs}_batch{args.batch_size}_op{optimizer}_test"

    # Load dataset
    dataset = "datav10.yaml"

    # Initialize model
    model = YOLOv10.from_pretrained('jameslahm/yolov10s')
    # Start the emissions tracker
    tracker = EmissionsTracker(log_level="WARNING")
    try:
        tracker.start()
    except Exception as e:
        print(f"Error starting emissions tracker: {e}")

    # Train model
    model.train(
        task='detect',
        cache=True,
        name=train_name,
        data=dataset,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        optimizer=optimizer,
        project=project,
        plots=True
    )

    # Validate model
    model.val(
        task='detect',
        name=val_name,
        data=dataset,
        imgsz=640,
        batch=batch_size,
        project=project,
        plots=True,
        save_json=True
    )



    # Test model
    model.val(
        task='detect',
        name=test_name,
        split='test',
        project=project,
        imgsz=640,
        batch=batch_size,
        data=dataset,
        plots=True,
        save_json=True
    )

    try:
        emissions_data = tracker.stop()
        # Define the output directory for the emissions data
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
