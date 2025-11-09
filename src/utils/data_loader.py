"""
Data Loading and Preprocessing Utilities

This module handles:
- Loading simulation data from iFogSim CSV exports
- Normalizing features for neural network training
- Creating train/val/test splits
- Handling time-series data windowing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTDataLoader:
    """
    Loads and preprocesses IoT simulation data for RL training.
    
    Expected CSV format from iFogSim:
    timestamp, node_id, cpu_util, mem_util, energy, latency, bandwidth, queue_len
    """
    
    def __init__(
        self,
        data_path: str,
        scaler_type: str = "standard",
        test_split: float = 0.2,
        val_split: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            data_path: Path to CSV file from simulation
            scaler_type: 'standard' (z-score) or 'minmax' (0-1 range)
            test_split: Fraction of data for test set
            val_split: Fraction of data for validation set
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.scaler_type = scaler_type
        self.test_split = test_split
        self.val_split = val_split
        self.seed = seed
        
        # Feature scalers (fit on train, transform on val/test)
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Data splits
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        # Metadata
        self.node_ids: list = []
        self.feature_names: list = []
        
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data and create train/val/test splits.
        
        Returns:
            (train_df, val_df, test_df)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Validate expected columns
        expected_cols = ['timestamp', 'node_id', 'cpu_util', 'mem_util', 
                        'energy', 'latency', 'bandwidth', 'queue_len']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"CSV must contain columns: {expected_cols}")
        
        # Store metadata
        self.node_ids = sorted(df['node_id'].unique().tolist())
        self.feature_names = [c for c in expected_cols if c not in ['timestamp', 'node_id']]
        
        logger.info(f"Loaded {len(df)} records for {len(self.node_ids)} nodes")
        logger.info(f"Time range: {df['timestamp'].min():.1f}s - {df['timestamp'].max():.1f}s")
        
        # Split data temporally (critical for time-series)
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        n = len(df_sorted)
        
        test_idx = int(n * (1 - self.test_split))
        val_idx = int(test_idx * (1 - self.val_split))
        
        self.train_data = df_sorted.iloc[:val_idx].copy()
        self.val_data = df_sorted.iloc[val_idx:test_idx].copy()
        self.test_data = df_sorted.iloc[test_idx:].copy()
        
        logger.info(f"Split: train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")
        
        # Fit scalers on training data only
        self._fit_scalers()
        
        # Apply normalization
        self.train_data = self._transform_features(self.train_data)
        self.val_data = self._transform_features(self.val_data)
        self.test_data = self._transform_features(self.test_data)
        
        return self.train_data, self.val_data, self.test_data
    
    def _fit_scalers(self):
        """Fit scalers on training data features."""
        for feature in self.feature_names:
            if self.scaler_type == "standard":
                scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
            
            scaler.fit(self.train_data[[feature]])
            self.scalers[feature] = scaler
        
        logger.info(f"Fitted {len(self.scalers)} feature scalers ({self.scaler_type})")
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to normalize features."""
        df_scaled = df.copy()
        for feature in self.feature_names:
            if feature in self.scalers:
                df_scaled[feature] = self.scalers[feature].transform(df[[feature]])
        return df_scaled
    
    def get_snapshot(self, timestamp: float, split: str = "train") -> Dict[str, np.ndarray]:
        """
        Get network snapshot at a specific timestamp.
        
        Args:
            timestamp: Simulation time
            split: 'train', 'val', or 'test'
        
        Returns:
            Dict with 'features' (N x F array) and 'node_ids' (N array)
        """
        data = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[split]
        
        # Get all nodes at this timestamp
        snapshot = data[data['timestamp'] == timestamp]
        
        if len(snapshot) == 0:
            raise ValueError(f"No data found at timestamp {timestamp}")
        
        # Extract feature matrix
        features = snapshot[self.feature_names].values
        node_ids = snapshot['node_id'].values
        
        return {
            "features": features,
            "node_ids": node_ids,
            "timestamp": timestamp
        }
    
    def get_time_window(
        self, 
        start_time: float, 
        end_time: float, 
        split: str = "train"
    ) -> pd.DataFrame:
        """
        Get data within a time window.
        
        Args:
            start_time: Window start timestamp
            end_time: Window end timestamp
            split: Data split to use
        
        Returns:
            DataFrame with all records in [start_time, end_time]
        """
        data = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[split]
        return data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
    
    def get_node_trajectory(self, node_id: str, split: str = "train") -> pd.DataFrame:
        """Get time-series trajectory for a specific node."""
        data = {"train": self.train_data, "val": self.val_data, "test": self.test_data}[split]
        return data[data['node_id'] == node_id].sort_values('timestamp')
    
    def inverse_transform(self, features: np.ndarray, feature_names: list) -> np.ndarray:
        """
        Convert normalized features back to original scale.
        
        Args:
            features: Normalized feature array (N x F)
            feature_names: List of feature names corresponding to columns
        
        Returns:
            Original-scale features
        """
        features_original = features.copy()
        for i, feature in enumerate(feature_names):
            if feature in self.scalers:
                features_original[:, i] = self.scalers[feature].inverse_transform(
                    features[:, [i]]
                ).flatten()
        return features_original


def create_time_windows(
    data: pd.DataFrame,
    window_size: int = 10,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding time windows for sequence models.
    
    Args:
        data: Time-series data (T x F)
        window_size: Number of timesteps per window
        stride: Step size between windows
    
    Returns:
        Windowed data (num_windows x window_size x F)
    """
    windows = []
    values = data.values
    
    for i in range(0, len(values) - window_size + 1, stride):
        window = values[i:i + window_size]
        windows.append(window)
    
    return np.array(windows)


def compute_statistics(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Compute summary statistics for features.
    
    Args:
        df: Input dataframe
        features: List of feature columns
    
    Returns:
        DataFrame with mean, std, min, max for each feature
    """
    stats = []
    for feature in features:
        stats.append({
            'feature': feature,
            'mean': df[feature].mean(),
            'std': df[feature].std(),
            'min': df[feature].min(),
            'max': df[feature].max(),
            'median': df[feature].median()
        })
    return pd.DataFrame(stats)


if __name__ == "__main__":
    # Example usage
    loader = IoTDataLoader(
        data_path="data/raw/sim_results.csv",
        scaler_type="standard",
        test_split=0.2,
        val_split=0.1
    )
    
    train_df, val_df, test_df = loader.load()
    
    # Print statistics
    print("\n=== Training Data Statistics ===")
    print(compute_statistics(train_df, loader.feature_names))
    
    # Get snapshot example
    first_timestamp = train_df['timestamp'].iloc[0]
    snapshot = loader.get_snapshot(first_timestamp, split="train")
    print(f"\nSnapshot at t={first_timestamp}: {snapshot['features'].shape}")
