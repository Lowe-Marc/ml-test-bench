import matplotlib.pyplot as plt
import numpy as np

def visualize_data(data, anomalies):
    """
    Plot multiple (data, anomalies) pairs on the same figure.
    Args:
        *series: Each argument should be a tuple (data, anomalies)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Data')
    anomaly_indices = np.where(anomalies == 1)[0]
    plt.scatter(anomaly_indices, data[anomaly_indices], color='red', label='Anomalies')
    plt.legend()
    plt.show()
