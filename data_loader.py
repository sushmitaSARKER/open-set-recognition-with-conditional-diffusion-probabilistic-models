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

def prepare_train_and_threshold_loaders(batch_size, train_ratio=0.8):
    """
    Prepares loaders for training (80%) and thresholding (20% + unknowns).
    """
    print("--- Preparing Train and Threshold DataLoaders with 80/20 Split ---")
    
    # 1. Load and process the two main data files
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1'])
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2'])
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("Main data files could not be loaded. Please check paths in config.py.")

    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    signals_np = combined_data[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(combined_data[config.DATA_LEN:, :]), axis=0).astype(int)
    
    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.zeros(len(labels), dtype=torch.long) # Placeholder for SNRs

    # 2. Get indices for all known and known-unknown classes
    known_indices_all = torch.cat([torch.where(labels == i)[0] for i in range(len(config.KNOWN_CLASSES_LIST))])
    known_unknown_indices = torch.where(labels == len(config.KNOWN_CLASSES_LIST))[0]
    
    # 3. Perform the 80/20 split ON THE KNOWN CLASSES ONLY
    shuffled_indices = known_indices_all[torch.randperm(len(known_indices_all))]
    split_idx = int(train_ratio * len(shuffled_indices))
    train_indices = shuffled_indices[:split_idx]
    valid_known_indices = shuffled_indices[split_idx:]
    
    # 4. Create the train_loader using the 80% split
    train_dataset = SignalDataset(signals[train_indices], labels[train_indices], snrs_tensor[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 5. Create the threshold_loader using the 20% validation split of knowns + all "known unknowns"
    thresh_indices = torch.cat([valid_known_indices, known_unknown_indices])
    threshold_dataset = SignalDataset(signals[thresh_indices], labels[thresh_indices], snrs_tensor[thresh_indices])
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created with correct 80/20 split.")
    print(f"  - Train Loader: {len(train_loader.dataset)} samples")
    print(f"  - Threshold Loader: {len(threshold_loader.dataset)} samples")
    
    return train_loader, threshold_loader

def prepare_test_loader(batch_size):
    """
    Prepares the final test loader from separate known and unknown test files.
    """
    print("\n--- Preparing Final Test DataLoader ---")
    
    # Load and process the separate known and unknown test files
    known_test_data = load_and_process_single_file(config.TEST_DATA_FILE_PATHS['known'])
    unknown_test_data = load_and_process_single_file(config.TEST_DATA_FILE_PATHS['unknown'])
    
    if known_test_data is None or unknown_test_data is None:
        raise FileNotFoundError("Test data files could not be loaded. Please check paths in config.py.")
        
    # Combine the known test and unknown test data
    test_data_combined = np.concatenate((known_test_data, unknown_test_data), axis=1)
    
    signals_np = test_data_combined[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(test_data_combined[config.DATA_LEN:, :]), axis=0).astype(int)
    
    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.zeros(len(labels), dtype=torch.long) # Placeholder for SNRs
    
    test_dataset = SignalDataset(signals, labels, snrs_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Test Loader created with {len(test_loader.dataset)} samples.")
    
    return test_loader
