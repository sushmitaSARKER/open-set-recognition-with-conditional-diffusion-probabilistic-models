import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from scipy.io import loadmat
import config

# --- HELPER FUNCTION TO PROCESS A SINGLE .MAT FILE ---
def load_and_process_single_file(file_path):
    """Loads, normalizes, and prepares data from a single .mat file."""
    print(f"  -> Processing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"    ERROR: File not found at {file_path}. Please check the path in config.py.")
        return None
        
    try:
        # Use scipy.io.loadmat for better compatibility with different .mat versions
        data_dict = loadmat(file_path)
        data = data_dict['vect']
    except Exception as e:
        print(f"    ERROR: Could not read file {file_path}. Details: {e}")
        return None

    # FIXED: Use config.DATA_LEN consistently
    signals = data[:config.DATA_LEN, :]  # Use 512, not hardcoded 500
    labels = data[config.DATA_LEN:, :]   # Start labels after DATA_LEN

    # Normalize signals
    mean = np.mean(signals, axis=0)
    # Use a small epsilon to avoid division by zero in case of flat signals
    std = (np.max(signals, axis=0) - np.min(signals, axis=0)) + 1e-8
    
    signals_normalized = (signals - mean) / std
    
    # Combine normalized signals with their original labels
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

# --- MAIN DATALOADER PREPARATION FUNCTIONS ---
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

    # 2. Create SNR labels for each file BEFORE combining
    snr_values_part1 = [5, 10, 15, 20, 25]
    num_samples_per_snr_part1 = data_part1.shape[1] // len(snr_values_part1)
    snr_labels_part1 = np.repeat(snr_values_part1, num_samples_per_snr_part1)

    snr_values_part2 = [-10, -5, 0]
    num_samples_per_snr_part2 = data_part2.shape[1] // len(snr_values_part2)
    snr_labels_part2 = np.repeat(snr_values_part2, num_samples_per_snr_part2)

    # Combine data and SNR labels separately
    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    combined_snrs = np.concatenate((snr_labels_part1, snr_labels_part2))
 
    # --- DEBUGGING LOGIC ---
    if config.DEBUG_MODE:
        print(f"!!! DEBUG MODE ON: Using a subset of {config.DEBUG_SUBSET_SIZE} samples for training/thresholding. !!!")
        p = np.random.permutation(combined_data.shape[1])
        subset_indices = p[:config.DEBUG_SUBSET_SIZE]
        combined_data = combined_data[:, subset_indices]
        combined_snrs = combined_snrs[subset_indices]
    # -------------------------
    
    # Extract signals and integer labels from the combined data
    signals_np = combined_data[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(combined_data[config.DATA_LEN:, :]), axis=0).astype(int)

    # ADD DEBUGGING TO IDENTIFY THE ISSUE:
    print(f"=== DATA DEBUGGING ===")
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Unique labels found: {np.unique(labels_int)}")
    print(f"Label counts: {dict(zip(*np.unique(labels_int, return_counts=True)))}")
    print(f"Expected known classes (0-{len(config.KNOWN_CLASSES_LIST)-1}): {list(range(len(config.KNOWN_CLASSES_LIST)))}")
    print(f"Expected known-unknown class: {len(config.KNOWN_CLASSES_LIST)}")
    print(f"======================")

    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.from_numpy(combined_snrs).to(torch.long)
    
    # 6. Get indices for all known and "known unknown" classes
    known_indices_all = torch.cat([torch.where(labels == i)[0] for i in range(len(config.KNOWN_CLASSES_LIST))])
    known_unknown_indices = torch.where(labels == len(config.KNOWN_CLASSES_LIST))[0] # e.g., Zigbee
    
    # ADD MORE DEBUGGING:
    print(f"Known class samples: {len(known_indices_all)}")
    print(f"Known-unknown class samples: {len(known_unknown_indices)}")
    
    # HANDLE MISSING UNKNOWN CLASS:
    if len(known_unknown_indices) == 0:
        print("❌ CRITICAL ERROR: No unknown class samples found!")
        print("Your data files don't contain the expected Zigbee class (label 3)")
        print("⚠️  Creating synthetic unknown samples for testing...")
        
        # Create synthetic unknown samples by adding noise to known samples
        num_synthetic = min(50, len(known_indices_all) // 10)  # 10% of known samples
        synthetic_indices = known_indices_all[torch.randperm(len(known_indices_all))[:num_synthetic]]
        
        # Add noise to signals and change labels to unknown class
        synthetic_signals = signals[synthetic_indices].clone()
        synthetic_signals += 0.1 * torch.randn_like(synthetic_signals)  # Add 10% noise
        synthetic_labels = torch.full((num_synthetic,), len(config.KNOWN_CLASSES_LIST), dtype=torch.long)
        synthetic_snrs = snrs_tensor[synthetic_indices]
        
        # Append synthetic data
        signals = torch.cat([signals, synthetic_signals])
        labels = torch.cat([labels, synthetic_labels])
        snrs_tensor = torch.cat([snrs_tensor, synthetic_snrs])
        
        # Update indices
        known_unknown_indices = torch.arange(len(signals) - num_synthetic, len(signals))
        print(f"Created {num_synthetic} synthetic unknown samples.")
    
    # 7. Perform the 80/20 split ON THE KNOWN CLASSES ONLY
    shuffled_indices = known_indices_all[torch.randperm(len(known_indices_all))]
    split_idx = int(train_ratio * len(shuffled_indices))
    train_indices = shuffled_indices[:split_idx]
    valid_known_indices = shuffled_indices[split_idx:]
    
    # 8. Create the train_loader using the 80% split of known classes
    train_dataset = SignalDataset(signals[train_indices], labels[train_indices], snrs_tensor[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 9. Create the threshold_loader by combining the 20% of knowns with ALL of the "known unknowns"
    thresh_indices = torch.cat([valid_known_indices, known_unknown_indices])
    threshold_dataset = SignalDataset(signals[thresh_indices], labels[thresh_indices], snrs_tensor[thresh_indices])
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created with correct 80/20 split and SNR labels.")
    print(f"  - Train Loader: {len(train_loader.dataset)} samples")
    print(f"  - Threshold Loader: {len(threshold_loader.dataset)} samples (Contains knowns and unknowns)")
    
    return train_loader, threshold_loader

def prepare_test_loader(batch_size):
    """
    Prepares the final test loader from separate known and unknown test files.
    """
    print("\n--- Preparing Final Test DataLoader ---")
    
    known_test_data = load_and_process_single_file(config.TEST_DATA_FILE_PATHS['known'])
    unknown_test_data = load_and_process_single_file(config.TEST_DATA_FILE_PATHS['unknown'])
    
    if known_test_data is None or unknown_test_data is None:
        raise FileNotFoundError("Test data files could not be loaded. Please check paths in config.py.")
    
    # --- DEBUGGING LOGIC ---
    if config.DEBUG_MODE:
        print(f"!!! DEBUG MODE ON: Using a subset of test samples. !!!")
        known_test_data = known_test_data[:, :config.DEBUG_SUBSET_SIZE // 2]
        unknown_test_data = unknown_test_data[:, :config.DEBUG_SUBSET_SIZE // 2]
    # -------------------------
        
    test_data_combined = np.concatenate((known_test_data, unknown_test_data), axis=1)
    
    signals_np = test_data_combined[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(test_data_combined[config.DATA_LEN:, :]), axis=0).astype(int)
    
    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.zeros(len(labels), dtype=torch.long) 
    
    test_dataset = SignalDataset(signals, labels, snrs_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Test Loader created with {len(test_loader.dataset)} samples.")
    
    return test_loader
