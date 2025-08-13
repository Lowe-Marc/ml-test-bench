

import pandas as pd
import numpy as np
from pathlib import Path

class DataReader:
    def __init__(self, data_path: str, batch_size: int = 1000):
        self.data_path = data_path
        self.batch_size = batch_size
        
        # State management
        self.parquet_files = []
        self.current_file_index = 0
        self.current_file_data = None
        self.current_position = 0
        self.is_initialized = False
    
    def _initialize(self):
        """Initialize the reader by finding all parquet files."""
        data_path = Path(self.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Get list of parquet files
        if data_path.is_dir():
            self.parquet_files = sorted(list(data_path.glob("**/*.parquet")))
            if not self.parquet_files:
                raise ValueError(f"No parquet files found in {data_path}")
        else:
            self.parquet_files = [data_path]
        
        print(f"Found {len(self.parquet_files)} parquet files")
        self.is_initialized = True
    
    def _load_current_file(self):
        """Load the current parquet file into memory."""
        if self.current_file_index >= len(self.parquet_files):
            return False  # No more files
        
        current_file = self.parquet_files[self.current_file_index]
        print(f"Loading file {self.current_file_index + 1}/{len(self.parquet_files)}: {current_file.name}")
        
        # Read parquet file
        df = pd.read_parquet(current_file)
        
        # Sort by index if it exists (for time series data)
        if 'index' in df.columns:
            df = df.sort_values('index')
        
        # Convert to numpy array
        if 'value' in df.columns:
            # Time series data - extract just the values
            self.current_file_data = df['value'].values
        else:
            # Generic data - convert all numeric columns to numpy
            self.current_file_data = df.select_dtypes(include=[np.number]).values.flatten()
        
        self.current_position = 0
        return True
    
    def read(self):
        """
        Return the next batch of data from the parquet file as a numpy array.

        Implementation notes:
        - Retrieve list of parquet files
        - Fill a batch (of self.batch_size) from the parquet files
            - If the current file is exhausted, move to the next file, but keep fulling the batch
        - Return the batch as a numpy array
        """
        if not self.is_initialized:
            self._initialize()
        
        # Initialize first file if needed
        if self.current_file_data is None:
            if not self._load_current_file():
                return None  # No data available
        
        batch = []
        remaining_batch_size = self.batch_size
        
        while remaining_batch_size > 0:
            # Check if current file is exhausted
            if self.current_position >= len(self.current_file_data):
                # Move to next file
                self.current_file_index += 1
                if not self._load_current_file():
                    # No more files, return what we have
                    break
            
            # Take data from current file
            available_in_file = len(self.current_file_data) - self.current_position
            to_take = min(remaining_batch_size, available_in_file)
            
            batch.extend(self.current_file_data[self.current_position:self.current_position + to_take])
            self.current_position += to_take
            remaining_batch_size -= to_take
        
        if not batch:
            return None  # No more data
        
        return np.array(batch)
    
    def reset(self):
        """Reset the reader to start from the beginning."""
        self.current_file_index = 0
        self.current_file_data = None
        self.current_position = 0
