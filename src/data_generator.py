import numpy as np
import os
import json

# TODO: Abstract this? It's duplicated
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")

class DataGenerator:
    """
    Class to generate synthetic data.
    """
    def __init__(self, data_size: int, num_anomalies: int, seed: int = None):
        self.data_size = data_size
        self.num_anomalies = num_anomalies
        self.seed = seed
    
    def generate_sinusoidal(self, save_data: bool = True):
        print("Generating data")
        if self.seed is not None:
            np.random.seed(self.seed)
        
        anomaly_ratio = self.num_anomalies / self.data_size
        
        # Create time steps for the sinusoidal wave
        time_steps = np.arange(self.data_size)
        
        # Generate base sinusoidal data with noise
        sinusoidal_data = np.sin(0.02 * time_steps) + np.random.normal(0, 0.1, size=self.data_size)
        
        # Generate anomaly labels (0 = normal, 1 = anomaly)
        anomaly_labels = np.zeros(self.data_size)
        anomaly_indices = np.random.choice(self.data_size, int(self.data_size * anomaly_ratio), replace=False)
        
        # Inject anomalies by adding large deviations to the data
        sinusoidal_data[anomaly_indices] += np.random.normal(3, 1, size=len(anomaly_indices))
        anomaly_labels[anomaly_indices] = 1
        
        if save_data:
            self._save_data(sinusoidal_data, anomaly_labels)
        return sinusoidal_data, anomaly_labels

    def _save_data(self, data, anomalies):
        """
        Save the generated data to a file in a new numbered environment folder under 'environments'.
        """
        env_dir = os.path.join(RAW_DATA_DIR, "generated", "sinusoidal")
        os.makedirs(env_dir, exist_ok=True)

        # Find next data file number
        existing = [d for d in os.listdir(env_dir) if d.startswith("data_")]
        if existing:
            nums = []
            for data_file in existing:
                num = data_file.split("_")[1].split(".")[0]
                if num.isdigit():
                    nums.append(int(num))
            data_num = max(nums) + 1
        else:
            data_num = 1

        data_path = os.path.join(env_dir, f"data_{data_num}.npy")
        np.save(data_path, data)
        print(f"Environment {data_num} saved to {data_path}")
        return data_path
