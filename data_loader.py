import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from scipy.io import loadmat
import config

# --- HELPER FUNCTION TO PROCESS A SINGLE .MAT FILE ---
def load_and_process_single_file(file_path):
    """Loads and normalizes data from a single .mat file."""
    print(f"  -> Processing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"    ERROR: File not found at {file_path}. Please check the path in config.py.")
        return None
        
    try:
        data_dict = loadmat(file_path)
        data = data_dict['vect']
    except Exception as e:
        print(f"    ERROR: Could not read file {file_path}. Details: {e}")
        return None

    signals = data[:config.DATA_LEN, :]
    labels = data[config.DATA_LEN:, :] 

    mean = np.mean(signals, axis=0)
    std = (np.max(signals, axis=0) - np.min(signals, axis=0)) + 1e-8
    
    signals_normalized = (signals - mean) / std
    
    processed_data = np.concatenate((signals_normalized, labels), axis=0)
    return processed_data

# --- DATASET CLASS ---
class SignalDataset(Dataset):
    """Custom PyTorch Dataset to handle signals, labels, and SNRs."""
    def __init__(self, signals, labels, snrs):
        self.signals = signals
        self.labels = labels
        self.snrs = snrs

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Return all three items for each sample
        return self.signals[idx], self.labels[idx], self.snrs[idx]

# --- MAIN DATALOADER PREPARATION FUNCTION ---

def prepare_dataloaders(batch_size, train_ratio=0.8):
    """
    Main function to load all data, combine it, split it, and create DataLoaders.
    This version correctly implements the 80/20 train/validation split.
    """
    print("--- Preparing DataLoaders with correct 80/20 split ---")
    
    # 1. Load and process both files
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1'])
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2'])
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("One or more data files could not be loaded. Please check paths in config.py.")

    # 2. Combine data from both files
    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    
    # 3. Shuffle all the samples (columns) randomly
    p = np.random.permutation(combined_data.shape[1])
    combined_data = combined_data[:, p]
    
    # 4. Perform the 80/20 train/validation split on the entire combined dataset
    num_samples = combined_data.shape[1]
    split_idx = int(train_ratio * num_samples)
    
    train_data_np = combined_data[:, :split_idx]
    valid_data_np = combined_data[:, split_idx:] # This is the remaining 20%

    # 5. Convert to PyTorch Tensors and Transpose
    train_tensor = torch.from_numpy(train_data_np.T).to(torch.complex64)
    valid_tensor = torch.from_numpy(valid_data_np.T).to(torch.complex64)
    
    # 6. Create Datasets and DataLoaders
    # The training set contains only a mix of known and unknown signals from the 80% split
    train_signals = train_tensor[:, :config.DATA_LEN]
    train_labels = torch.argmax(torch.abs(train_tensor[:, config.DATA_LEN:]), dim=1)
    train_snrs = torch.zeros(len(train_labels)) # Placeholder for SNR if needed later
    train_dataset = SignalDataset(train_signals, train_labels, train_snrs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # The validation 20% will be used for both thresholding and final testing
    valid_signals = valid_tensor[:, :config.DATA_LEN]
    valid_labels = torch.argmax(torch.abs(valid_tensor[:, config.DATA_LEN:]), dim=1)
    valid_snrs = torch.zeros(len(valid_labels)) # Placeholder
    
    # Create threshold and test sets from the validation split
    known_mask = valid_labels < len(config.KNOWN_CLASSES_LIST)
    known_unknown_mask = valid_labels == len(config.KNOWN_CLASSES_LIST) # Zigbee
    test_unknown_mask = valid_labels == len(config.KNOWN_CLASSES_LIST) + 1 # DSSS

    # Threshold set = Knowns + Zigbee from the validation split
    thresh_indices = torch.where(known_mask | known_unknown_mask)[0]
    threshold_dataset = SignalDataset(valid_signals[thresh_indices], valid_labels[thresh_indices], valid_snrs[thresh_indices])
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)

    # Test set = Knowns + DSSS from the validation split
    test_indices = torch.where(known_mask | test_unknown_mask)[0]
    test_dataset = SignalDataset(valid_signals[test_indices], valid_labels[test_indices], valid_snrs[test_indices])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoaders created successfully with correct splits:")
    print(f"  - Train Loader: {len(train_loader.dataset)} samples")
    print(f"  - Threshold Loader: {len(threshold_loader.dataset)} samples")
    print(f"  - Test Loader: {len(test_loader.dataset)} samples")

    return train_loader, threshold_loader, test_loader
