import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import json
import re


from anomaly_detection.src.data_generator import DataGenerator
from anomaly_detection.src.trainer import Trainer
from anomaly_detection.src.visualization import visualize_data
from anomaly_detection.src.model import LSTMAutoencoder

data_generation_config = {
    "data_size": 500,
    "num_anomalies": 1,
    "num_features_per_timestep": 1,
    "training_seed": None,
    "training_anomaly_ratio": 0,
}

hyperparameters = {
    "num_training_iterations": 500,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "hidden_size": 64
}

training_batches = []
anomaly_ratio = data_generation_config["num_anomalies"]
for i in range(hyperparameters["num_training_iterations"]):
    print(f"Generating training data batch {i + 1} with {data_generation_config['num_anomalies']} anomalies ({anomaly_ratio:.2%} anomaly ratio)")
    training_data, training_anomalies = generate_timeseries(data_size, training_anomaly_ratio, seed=training_seed)
    training_batches.append((training_data, training_anomalies))
    # visualize_data(training_data, training_anomalies)



model = LSTMAutoencoder(num_features_per_timestep, hyperparameters["hidden_size"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])


# Inference on a non-training batch
print("\nEvaluating on a new (non-training) batch:")
test_data, test_anomalies = generate_timeseries(length=data_size, anomaly_ratio=1/1000, seed=None)
test_batch = torch.from_numpy(test_data).float().unsqueeze(0)
model.eval()
with torch.no_grad():
    test_prediction = model(test_batch)
    test_prediction_np = test_prediction.squeeze(0).cpu().numpy()
    test_original_np = test_batch.squeeze(0).cpu().numpy()


# Plot original vs. reconstructed
plt.figure(figsize=(12, 6))
plt.plot(test_original_np, label='Original (Test)')
plt.plot(test_prediction_np, label='Reconstruction (Test)', linestyle='dashed')
plt.legend()
plt.title('Original vs. Reconstructed on Non-Training Batch')
plt.show()

# Compute and plot reconstruction error
reconstruction_error = np.abs(test_original_np - test_prediction_np).flatten()
anomaly_indices = np.where(test_anomalies == 1)[0]

plt.figure(figsize=(12, 4))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.scatter(anomaly_indices, reconstruction_error[anomaly_indices], color='red', label='Anomalies', zorder=5)
plt.xlabel('Time Step')
plt.ylabel('Absolute Error')
plt.title('Reconstruction Error (Anomalies Highlighted)')
plt.legend()
plt.show()


