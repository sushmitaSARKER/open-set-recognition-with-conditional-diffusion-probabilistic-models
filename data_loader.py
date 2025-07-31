import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import h5py
import config

# --- HELPER FUNCTION TO PROCESS A SINGLE .MAT FILE ---
def load_and_process_single_file(file_path):
    """Loads and normalizes data from a single .mat file."""
    print(f"  -> Processing file: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['vect'][:]
    except Exception as e:
        print(f"    ERROR: Could not read file {file_path}. Details: {e}")
        return None

    # As per previous code, signals are the first 500 rows
    signals = data[:500, :]
    labels = data[500:, :] # Keep all label rows (e.g., 500:504)

    # Normalize signalssss
    mean = np.mean(signals, axis=0)
    std = (np.max(signals, axis=0) - np.min(signals, axis=0)) + 1e-8
    
    signals_normalized = (signals - mean) / std
    
    # Combine normalized signals with their original labels
    processed_data = np.concatenate((signals_normalized, labels), axis=0)
    return processed_data

# --- MAIN DATALOADER PREPARATION FUNCTION ---
def prepare_dataloaders(batch_size, train_ratio=0.8):
    print("--- Preparing DataLoaders ---")
    
    # Load and process both files
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1'])
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2'])
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("One or more data files could not be loaded. Please check paths in config.py.")

    # sCombine data from both files by concatenating samples (horizontally, axis=1)
    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    
    # Shuffle all the samples randomly
    combined_data = combined_data[:, np.random.permutation(combined_data.shape[1])]
    
    # Split into training and validation sets
    num_samples = combined_data.shape[1]
    split_idx = int(train_ratio * num_samples)
    
    train_data_np = combined_data[:, :split_idx]
    valid_data_np = combined_data[:, split_idx:]

    print(f"Combined data into {train_data_np.shape[1]} training samples and {valid_data_np.shape[1]} validation samples.")

    # Convert to PyTorch Tensors and Transpose
    # The final shape should be [num_samples, num_features]
    train_tensor = torch.from_numpy(train_data_np.T).to(torch.complex64)
    valid_tensor = torch.from_numpy(valid_data_np.T).to(torch.complex64)
    
    print(f"Final training tensor shape: {train_tensor.shape}")
    print(f"Final validation tensor shape: {valid_tensor.shape}")

    # Create Datasets and DataLoaders
    # The dataset will now hold the full [504] vector for each sample
    train_dataset = TensorDataset(train_tensor)
    valid_dataset = TensorDataset(valid_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully.")
    return train_loader, validation_loader