import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from src.data_generator import DataGenerator
from src.trainer import Trainer
from src.model import LSTMAutoencoder
from src.transform import transform_npy_to_parquet
from src.data_reader import DataReader

HIDDEN_SIZE = 64

def load_json_config(path):
    if not os.path.exists(path):
        print(f"Config file {path} not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)

def generate_data():
    generator = DataGenerator(data_size=1000, num_anomalies=10)
    generator.generate_sinusoidal()

def transform_data():
    data_path = "raw_data/generated/sinusoidal"
    output_path = "processed_data/generated/sinusoidal"
    transform_npy_to_parquet(data_path, output_path)

# TODO: Lookup experiment path, don't use config, include description
def run_training():
    num_epochs = 500
    learning_rate = 1e-3
    batch_size = 100
    training_data_path = "processed_data/generated/sinusoidal"
    data_reader = DataReader(training_data_path, batch_size=batch_size)

    model = LSTMAutoencoder(input_size=1, hidden_size=HIDDEN_SIZE)
    trainer = Trainer(model=model, num_epochs=num_epochs, learning_rate=learning_rate, data_reader=data_reader)
    exp_num = trainer.run_training()
    return exp_num

# TODO: Change config to just take experiment number
def run_inference():
    experiment_path = "experiments/experiment_1/model.pt"

    model = LSTMAutoencoder(input_size=1, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(experiment_path))
    
    # Generate fresh test data (different from training data)
    print("Generating fresh test data for inference...")
    test_generator = DataGenerator(data_size=1000, num_anomalies=15, seed=9999)  # Different seed and anomaly count
    test_data, test_anomalies = test_generator.generate_sinusoidal(save_data=False)
    
    # Create inference output directory
    inference_dir = "inference"
    os.makedirs(inference_dir, exist_ok=True)
    
    model.eval()
    print("Running inference...")
    
    with torch.no_grad():
        # Convert test data to tensor and run inference
        test_tensor = torch.from_numpy(test_data).float().unsqueeze(-1)
        predictions = model(test_tensor)
        
        # Convert back to numpy
        predictions_np = predictions.squeeze(-1).cpu().numpy()
        reconstruction_error = np.abs(test_data - predictions_np)
    
    # Save inference results and original test data to parquet files
    original_df = pd.DataFrame({
        'index': range(len(test_data)),
        'value': test_data
    })
    
    predictions_df = pd.DataFrame({
        'index': range(len(predictions_np)),
        'value': predictions_np
    })
    
    reconstruction_error_df = pd.DataFrame({
        'index': range(len(reconstruction_error)),
        'reconstruction_error': reconstruction_error
    })
    
    # Also save anomaly labels for evaluation
    anomalies_df = pd.DataFrame({
        'index': range(len(test_anomalies)),
        'is_anomaly': test_anomalies.astype(bool)
    })
    
    original_path = os.path.join(inference_dir, "test_data.parquet")
    predictions_path = os.path.join(inference_dir, "predictions.parquet")
    error_path = os.path.join(inference_dir, "reconstruction_errors.parquet")
    anomalies_path = os.path.join(inference_dir, "anomaly_labels.parquet")
    
    original_df.to_parquet(original_path, index=False)
    predictions_df.to_parquet(predictions_path, index=False)
    reconstruction_error_df.to_parquet(error_path, index=False)
    anomalies_df.to_parquet(anomalies_path, index=False)
    
    print(f"Inference complete. Results saved to {inference_dir}")
    print(f"Test data: {original_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Reconstruction errors: {error_path}")
    print(f"Anomaly labels: {anomalies_path}")
    
    return inference_dir

def visualize(parquet_path=None):
    """
    Visualize time series data from parquet files.
    Can show original data, predictions, and reconstruction errors.
    """
    if parquet_path is None:
        # Default paths for comprehensive visualization
        original_path = "inference/test_data.parquet"  # Use fresh test data instead of training data
        predictions_path = "inference/predictions.parquet"
        errors_path = "inference/reconstruction_errors.parquet"
        anomalies_path = "inference/anomaly_labels.parquet"
        
        # Load original data
        original_data = None
        if os.path.exists(original_path):
            df_original = pd.read_parquet(original_path)
            if 'index' in df_original.columns and 'value' in df_original.columns:
                df_original = df_original.sort_values('index')
                original_data = df_original
                print(f"Loaded original data: {df_original.shape}")
        
        # Load predictions
        predictions_data = None
        if os.path.exists(predictions_path):
            df_predictions = pd.read_parquet(predictions_path)
            if 'index' in df_predictions.columns and 'value' in df_predictions.columns:
                df_predictions = df_predictions.sort_values('index')
                predictions_data = df_predictions
                print(f"Loaded predictions: {df_predictions.shape}")
        
        # Load reconstruction errors
        errors_data = None
        if os.path.exists(errors_path):
            df_errors = pd.read_parquet(errors_path)
            if 'index' in df_errors.columns and 'reconstruction_error' in df_errors.columns:
                df_errors = df_errors.sort_values('index')
                errors_data = df_errors
                print(f"Loaded reconstruction errors: {df_errors.shape}")
        
        # Load anomaly labels
        anomalies_data = None
        if os.path.exists(anomalies_path):
            df_anomalies = pd.read_parquet(anomalies_path)
            if 'index' in df_anomalies.columns and 'is_anomaly' in df_anomalies.columns:
                df_anomalies = df_anomalies.sort_values('index')
                anomalies_data = df_anomalies
                print(f"Loaded anomaly labels: {df_anomalies.shape}")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Original vs Reconstructed
        if original_data is not None and predictions_data is not None:
            axes[0].plot(original_data['index'], original_data['value'], label='Original', alpha=0.8)
            axes[0].plot(predictions_data['index'], predictions_data['value'], 
                        label='Reconstruction', linestyle='--', alpha=0.8)
            axes[0].set_title('Original vs Reconstructed Time Series')
            axes[0].set_xlabel('Time Step')
            axes[0].set_ylabel('Value')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        elif original_data is not None:
            axes[0].plot(original_data['index'], original_data['value'], label='Original')
            axes[0].set_title('Original Time Series')
            axes[0].set_xlabel('Time Step')
            axes[0].set_ylabel('Value')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Original vs Reconstructed Time Series')
        
        # Plot 2: Reconstruction Errors with Anomaly Highlights
        if errors_data is not None:
            axes[1].plot(errors_data['index'], errors_data['reconstruction_error'], 
                        label='Reconstruction Error', color='red', alpha=0.7)
            
            # Highlight true anomalies if available
            if anomalies_data is not None:
                anomaly_indices = anomalies_data[anomalies_data['is_anomaly']]['index']
                if len(anomaly_indices) > 0:
                    anomaly_errors = errors_data[errors_data['index'].isin(anomaly_indices)]['reconstruction_error']
                    axes[1].scatter(anomaly_indices, anomaly_errors, 
                                  color='orange', s=30, label='True Anomalies', zorder=5)
            
            axes[1].set_title('Reconstruction Error (True Anomalies Highlighted)')
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('Absolute Error')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No error data available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Reconstruction Error')
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Single file visualization (original behavior)
        if not os.path.exists(parquet_path):
            print(f"Parquet file not found: {parquet_path}")
            return
        
        df = pd.read_parquet(parquet_path)
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        plt.figure(figsize=(12, 6))
        
        if 'index' in df.columns and 'value' in df.columns:
            df = df.sort_values('index')
            plt.plot(df['index'], df['value'], label='Time Series Data')
            plt.ylabel('Value')
        elif 'index' in df.columns and 'reconstruction_error' in df.columns:
            df = df.sort_values('index')
            plt.plot(df['index'], df['reconstruction_error'], label='Reconstruction Error', color='red')
            plt.ylabel('Reconstruction Error')
        else:
            print(f"Expected columns 'index' and 'value' or 'reconstruction_error', but found: {df.columns.tolist()}")
            print("First few rows:")
            print(df.head())
            return
        
        plt.xlabel('Time Step')
        plt.title(f'Data Visualization: {os.path.basename(parquet_path)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection CLI/API (JSON-driven)")
    subparsers = parser.add_subparsers(dest="command")

    parser_env = subparsers.add_parser("generate_data", help="Generate a set of data")

    parser_transform = subparsers.add_parser("transform", help="Transform the data")

    parser_train = subparsers.add_parser("train", help="Train a model using training.json")

    parser_infer = subparsers.add_parser("inference", help="Run inference using inference.json")

    parser_visualize = subparsers.add_parser("visualize", help="Run visualization")
    parser_visualize.add_argument("--parquet_path", help="Path to specific parquet file to visualize (optional)")

    args = parser.parse_args()
    if args.command == "generate_data":
        generate_data()
    elif args.command == "transform":
        transform_data()
    elif args.command == "train":
        run_training()
    elif args.command == "inference":
        run_inference()
    elif args.command == "visualize":
        visualize()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
