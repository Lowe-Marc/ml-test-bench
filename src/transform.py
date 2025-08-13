import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

# TODO: Abstract this? It's duplicated
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")


def transform_csv_to_parquet():
    pass

def transform_npy_to_parquet(data_path: str, output_dir: str):
    """
    Read all npy files in the given directory and convert them to parquet format, 
    writing to the output_dir directory.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files in the directory
    npy_files = list(data_path.glob("**/*.npy"))
    if not npy_files:
        print(f"No .npy files found in {data_path}")
        return
    
    print(f"Found {len(npy_files)} .npy files to convert")
    
    for npy_file in npy_files:
        try:
            # Load numpy data
            data = np.load(npy_file)
            print(f"Processing {npy_file.name}, shape: {data.shape}")
            
            # Convert to DataFrame - create separate record for each entry
            if data.ndim == 1:
                # 1D array: create records with index and value
                df = pd.DataFrame({
                    'index': range(len(data)),
                    'value': data
                })
            elif data.ndim == 2:
                # TODO: Abstract this to arbitrary dimensions
                # 2D array: flatten and create records with row, column, and value
                records = []
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        records.append({
                            'row': i,
                            'column': j,
                            'value': data[i, j]
                        })
                df = pd.DataFrame(records)
            else:
                print(f"Skipping {npy_file.name}: unsupported dimensionality {data.ndim}")
                continue
            
            # Create output filename maintaining subdirectory structure
            relative_path = npy_file.relative_to(data_path)
            output_path = output_dir / relative_path.with_suffix('.parquet')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            df.to_parquet(output_path, index=False)
            print(f"Saved {output_path}")
            
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
    
    print(f"Conversion complete. Files saved to {output_dir}")
