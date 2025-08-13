import numpy as np
import os
import json

class DataGenerator:
    """
    Class to generate synthetic data.
    """
    def __init__(self, num_partitions: int, data_size: int, num_anomalies: int, seed: int = None):
        self.num_partitions = num_partitions
        self.data_size = data_size
        self.num_anomalies = num_anomalies
        self.seed = seed
    
    def generate_sinusoidal(self, save_data: bool = True):
        print("Generating data")
        anomaly_ratio = self.num_anomalies / self.data_size
        all_data = []
        all_anomalies = []
        for i in range(self.num_partitions):
            if self.seed is not None:
                np.random.seed(self.seed + i)  # ensure different partitions if seed is set
            else:
                np.random.seed()
            t = np.arange(self.data_size)
            base = np.sin(0.02 * t) + np.random.normal(0, 0.1, size=self.data_size)
            anomalies = np.zeros(self.data_size)
            anomaly_indices = np.random.choice(self.data_size, int(self.data_size * anomaly_ratio), replace=False)
            base[anomaly_indices] += np.random.normal(3, 1, size=len(anomaly_indices))
            anomalies[anomaly_indices] = 1
            all_data.append(base)
            all_anomalies.append(anomalies)
        all_data = np.stack(all_data)  # shape: (num_partitions, data_size)
        all_anomalies = np.stack(all_anomalies)  # shape: (num_partitions, data_size)
        if save_data:
            self._save_environment(all_data, all_anomalies)
        return all_data, all_anomalies

    def _save_environment(self, data, anomalies):
        """
        Save the generated data to a file in a new numbered environment folder under 'environments'.
        """
        env_dir = "environments"
        os.makedirs(env_dir, exist_ok=True)

        # Find next environment number
        existing = [d for d in os.listdir(env_dir) if d.startswith("env_")]
        if existing:
            nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
            env_num = max(nums) + 1
        else:
            env_num = 1

        env_path = os.path.join(env_dir, f"env_{env_num}")
        os.makedirs(env_path, exist_ok=True)

        # Save data and anomalies
        np.save(os.path.join(env_path, "data.npy"), data)
        np.save(os.path.join(env_path, "anomalies.npy"), anomalies)

        # Save metadata
        meta = {
            "data_size": self.data_size,
            "num_anomalies": self.num_anomalies,
            "seed": self.seed
        }
        with open(os.path.join(env_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Environment {env_num} saved to {env_path}")
        return env_num
