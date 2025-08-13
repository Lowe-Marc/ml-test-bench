import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.data_generator import DataGenerator
from src.trainer import Trainer
from src.model import LSTMAutoencoder

HIDDEN_SIZE = 2000

def load_json_config(path):
    if not os.path.exists(path):
        print(f"Config file {path} not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)

def generate_environment(config_path):
    config = load_json_config(config_path)
    generator = DataGenerator(**config)
    generator.generate_sinusoidal()
    
# TODO: Lookup experiment path, don't use config, include description
def run_training(config_path):
    config = load_json_config(config_path)
    # Assumes Trainer class exists and can take config dict
    model = LSTMAutoencoder(input_size=1, hidden_size=HIDDEN_SIZE)
    trainer = Trainer(**config, model=model)
    exp_num = trainer.run_training()
    return exp_num

# TODO: Change config to just take experiment number
def run_inference(config_path):
    config = load_json_config(config_path)
    
    model = LSTMAutoencoder(input_size=1, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(config["experiment_path"]))

    # Visualization
    generator = DataGenerator(num_partitions=1, data_size=1000, num_anomalies=10)
    data, anomalies = generator.generate_sinusoidal(save_data=False)

    data_tensor = torch.from_numpy(data).float().unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        predictions = model(data_tensor)

    # Plot original vs. reconstructed
    plottable_original_data = data[0]
    plottable_predictions = predictions.squeeze(0).squeeze(-1)
    plt.figure(figsize=(12, 6))
    plt.plot(plottable_original_data, label='Original')
    plt.plot(plottable_predictions, label='Reconstruction', linestyle='dashed')
    plt.legend()
    plt.title('Original vs. Reconstructed')
    plt.show()

    # Reconstruction error
    predictions_np = predictions.squeeze(0).squeeze(-1).cpu().numpy()
    reconstruction_error = np.abs(data[0] - predictions_np)
    anomaly_indices = (anomalies[0] == 1).nonzero()[0]
    plt.figure(figsize=(12, 4))
    plt.plot(reconstruction_error, label='Reconstruction Error')
    plt.scatter(anomaly_indices, reconstruction_error[anomaly_indices], color='red', label='Anomalies', zorder=5)
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Reconstruction Error (Anomalies Highlighted)')
    plt.legend()
    plt.show()
    print("Inference and visualization complete.")

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection CLI/API (JSON-driven)")
    subparsers = parser.add_subparsers(dest="command")

    parser_env = subparsers.add_parser("generate_env", help="Generate a data environment from environment.json")
    parser_env.add_argument("--config", type=str, default="environment.json")

    parser_train = subparsers.add_parser("train", help="Train a model using training.json")
    parser_train.add_argument("--config", type=str, default="training.json")

    parser_infer = subparsers.add_parser("inference", help="Run inference using inference.json")
    parser_infer.add_argument("--config", type=str, default="inference.json")

    args = parser.parse_args()
    if args.command == "generate_env":
        generate_environment(args.config)
    elif args.command == "train":
        run_training(args.config)
    elif args.command == "inference":
        run_inference(args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
