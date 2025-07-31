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
def prepare_dataloaders(batch_size):
    """
    Main function to load all data, combine it, split it, and create DataLoaders.
    This version correctly handles the 8 SNR levels from your two data files.
    """
    print("--- Preparing DataLoaders (with 8 SNR levels) ---")
    
    # Load and process both files
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1']) # e.g., 5dB to 25dB
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2']) # e.g., -10dB to 0dB
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("One or more data files could not be loaded. Please check paths in config.py.")

    # --- CORRECTED SNR and Data Combination Logic ---
    # Create SNR labels for each file BEFORE combining and shuffling
    snr_values_part1 = [5, 10, 15, 20, 25]
    num_samples_per_snr_part1 = data_part1.shape[1] // len(snr_values_part1)
    snr_labels_part1 = np.repeat(snr_values_part1, num_samples_per_snr_part1)

    snr_values_part2 = [-10, -5, 0]
    num_samples_per_snr_part2 = data_part2.shape[1] // len(snr_values_part2)
    snr_labels_part2 = np.repeat(snr_values_part2, num_samples_per_snr_part2)

    # Combine data and SNR labels separately
    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    combined_snrs = np.concatenate((snr_labels_part1, snr_labels_part2))

    # Shuffle the data and SNR labels together to maintain correspondence
    p = np.random.permutation(combined_data.shape[1])
    combined_data = combined_data[:, p]
    combined_snrs = combined_snrs[p]
    
    # Extract signals and integer labels from the combined data
    signals_np = combined_data[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(combined_data[config.DATA_LEN:, :]), axis=0).astype(int)

    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.from_numpy(combined_snrs).to(torch.long)
    
    # Split data based on class labels
    known_indices = torch.cat([torch.where(labels == i)[0] for i in range(len(config.KNOWN_CLASSES_LIST))])
    known_unknown_indices = torch.where(labels == len(config.KNOWN_CLASSES_LIST))[0]
    test_unknown_indices = torch.where(labels == len(config.KNOWN_CLASSES_LIST) + 1)[0]
    
    # Create PyTorch Datasets
    train_dataset = SignalDataset(signals[known_indices], labels[known_indices], snrs_tensor[known_indices])
    
    thresh_indices = torch.cat([known_indices, known_unknown_indices])
    threshold_dataset = SignalDataset(signals[thresh_indices], labels[thresh_indices], snrs_tensor[thresh_indices])
    
    test_indices = torch.cat([known_indices, test_unknown_indices])
    test_dataset = SignalDataset(signals[test_indices], labels[test_indices], snrs_tensor[test_indices])
    
    # Create and return all three DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully.")
    print(f"  - Train Loader: {len(train_loader.dataset)} samples")
    print(f"  - Threshold Loader: {len(threshold_loader.dataset)} samples")
    print(f"  - Test Loader: {len(test_loader.dataset)} samples")

    return train_loader, threshold_loader, test_loader