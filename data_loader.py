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
        data_dict = loadmat(file_path)
        data = data_dict['vect']
    except Exception as e:
        print(f"    ERROR: Could not read file {file_path}. Details: {e}")
        return None

    signals = data[:config.DATA_LEN, :]
    labels = data[config.DATA_LEN:, :]

    # Normalize signals
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
        return self.signals[idx], self.labels[idx], self.snrs[idx]

# --- ZIGBEE DATA EXTRACTION FOR THRESHOLD CALCULATION ---
def prepare_zigbee_threshold_data():
    """Extract Zigbee data from the test 'known' file for threshold calculation"""
    zigbee_dsss_file = config.TEST_DATA_FILE_PATHS['known']
    zigbee_data = load_and_process_single_file(zigbee_dsss_file)
    
    if zigbee_data is None:
        print("WARNING: Could not load Zigbee data for threshold calculation")
        return None, None, None
    
    # Extract only Zigbee samples (first 3200 samples)
    zigbee_signals = zigbee_data[:config.DATA_LEN, :3200]  # First 3200 samples are Zigbee
    zigbee_labels = np.full(3200, 3)  # Label them as class 3 (known-unknown)
    zigbee_snrs = np.zeros(3200)
    
    print(f"Loaded {zigbee_signals.shape[1]} Zigbee samples for threshold calculation")
    
    return zigbee_signals, zigbee_labels, zigbee_snrs

# --- DSSS DATA EXTRACTION FOR TESTING ---
def prepare_dsss_test_data():
    """Extract DSSS data for testing"""
    # Load both files that contain DSSS data
    known_file = config.TEST_DATA_FILE_PATHS['known']  # Contains Zigbee + DSSS
    unknown_file = config.TEST_DATA_FILE_PATHS['unknown']  # Contains more DSSS
    
    known_data = load_and_process_single_file(known_file)
    unknown_data = load_and_process_single_file(unknown_file)
    
    if known_data is None or unknown_data is None:
        print("WARNING: Could not load DSSS data for testing")
        return None, None, None
    
    # Extract DSSS samples from both files
    dsss_from_known = known_data[:config.DATA_LEN, 3200:]  # DSSS samples from first file (after Zigbee)
    dsss_from_unknown = unknown_data[:config.DATA_LEN, :]  # All samples from second file
    
    # Combine DSSS data
    dsss_signals = np.concatenate([dsss_from_known, dsss_from_unknown], axis=1)
    num_dsss = dsss_signals.shape[1]
    
    dsss_labels = np.full(num_dsss, 4)  # Label as class 4 (unknown)
    dsss_snrs = np.zeros(num_dsss)
    
    print(f"Loaded {num_dsss} DSSS samples for testing")
    
    return dsss_signals, dsss_labels, dsss_snrs

# --- MAIN DATALOADER PREPARATION FUNCTIONS ---
def prepare_train_and_threshold_loaders(batch_size, train_ratio=0.8, threshold_ratio=0.1):
    """
    Prepares loaders following TRUE 8:1:1 ratio:
    - Training: 80% of closed set (LTE, BT LE, WLAN)
    - Threshold (validation): 10% of closed set + Zigbee samples
    - Testing: 10% of closed set + DSSS samples (prepared separately)
    """
    print("--- Preparing Train and Threshold DataLoaders with 8:1:1 Split ---")
    
    # Load closed set data (LTE, BT LE, WLAN)
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1'])
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2'])
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("Main data files could not be loaded. Please check paths in config.py.")

    # Create SNR labels
    snr_values_part1 = [5, 10, 15, 20, 25]
    num_samples_per_snr_part1 = data_part1.shape[1] // len(snr_values_part1)
    snr_labels_part1 = np.repeat(snr_values_part1, num_samples_per_snr_part1)

    snr_values_part2 = [-10, -5, 0]
    num_samples_per_snr_part2 = data_part2.shape[1] // len(snr_values_part2)
    snr_labels_part2 = np.repeat(snr_values_part2, num_samples_per_snr_part2)

    # Combine closed set data
    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    combined_snrs = np.concatenate((snr_labels_part1, snr_labels_part2))
 
    # Debug mode handling
    if config.DEBUG_MODE:
        print(f"!!! DEBUG MODE ON: Using a subset of {config.DEBUG_SUBSET_SIZE} samples !!!")
        p = np.random.permutation(combined_data.shape[1])
        subset_indices = p[:config.DEBUG_SUBSET_SIZE]
        combined_data = combined_data[:, subset_indices]
        combined_snrs = combined_snrs[subset_indices]
    
    # Extract signals and labels from closed set
    signals_np = combined_data[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(combined_data[config.DATA_LEN:, :]), axis=0).astype(int)

    print(f"=== CLOSED SET DATA (8:1:1 Split) ===")
    print(f"Total closed set samples: {combined_data.shape[1]}")
    print(f"Unique labels in closed set: {np.unique(labels_int, return_counts=True)}")
    print(f"======================================")

    # Convert closed set to tensors
    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.from_numpy(combined_snrs).to(torch.long)
    
    # Get indices for closed set classes
    known_indices_all = torch.cat([torch.where(labels == i)[0] for i in range(len(config.KNOWN_CLASSES_LIST))])
    
    # IMPLEMENT TRUE 8:1:1 SPLIT ON CLOSED SET
    shuffled_indices = known_indices_all[torch.randperm(len(known_indices_all))]
    
    train_split_idx = int(train_ratio * len(shuffled_indices))  # 80%
    threshold_split_idx = int((train_ratio + threshold_ratio) * len(shuffled_indices))  # 90%
    
    train_indices = shuffled_indices[:train_split_idx]  # 80% for training
    threshold_known_indices = shuffled_indices[train_split_idx:threshold_split_idx]  # 10% for threshold
    # test_known_indices = shuffled_indices[threshold_split_idx:]  # 10% for testing (used later)
    
    # Store the test indices for later use
    global test_known_indices_global
    test_known_indices_global = shuffled_indices[threshold_split_idx:]
    
    print(f"=== 8:1:1 SPLIT BREAKDOWN ===")
    print(f"Training samples: {len(train_indices)} ({len(train_indices)/len(shuffled_indices)*100:.1f}%)")
    print(f"Threshold samples: {len(threshold_known_indices)} ({len(threshold_known_indices)/len(shuffled_indices)*100:.1f}%)")
    print(f"Test samples reserved: {len(test_known_indices_global)} ({len(test_known_indices_global)/len(shuffled_indices)*100:.1f}%)")
    print(f"=============================")
    
    # ADD ZIGBEE DATA FOR THRESHOLD CALCULATION
    zigbee_signals, zigbee_labels, zigbee_snrs = prepare_zigbee_threshold_data()
    
    if zigbee_signals is not None:
        print("Adding real Zigbee data for threshold calculation...")
        
        zigbee_signals_tensor = torch.from_numpy(zigbee_signals.T).to(torch.complex64)
        zigbee_labels_tensor = torch.from_numpy(zigbee_labels).to(torch.long)
        zigbee_snrs_tensor = torch.from_numpy(zigbee_snrs).to(torch.long)
        
        # Append Zigbee data to existing data
        signals = torch.cat([signals, zigbee_signals_tensor])
        labels = torch.cat([labels, zigbee_labels_tensor])
        snrs_tensor = torch.cat([snrs_tensor, zigbee_snrs_tensor])
        
        # Update Zigbee indices
        zigbee_indices = torch.arange(len(signals) - len(zigbee_signals_tensor), len(signals))
        
        print(f"Added {len(zigbee_signals_tensor)} Zigbee samples for threshold calculation")
    
    # Create train loader (80% closed set only)
    train_dataset = SignalDataset(signals[train_indices], labels[train_indices], snrs_tensor[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create threshold loader (10% closed set + all Zigbee)
    if zigbee_signals is not None:
        thresh_indices = torch.cat([threshold_known_indices, zigbee_indices])
    else:
        thresh_indices = threshold_known_indices
        
    threshold_dataset = SignalDataset(signals[thresh_indices], labels[thresh_indices], snrs_tensor[thresh_indices])
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created with proper 8:1:1 split.")
    print(f"  - Train Loader: {len(train_loader.dataset)} samples (80% closed set)")
    print(f"  - Threshold Loader: {len(threshold_loader.dataset)} samples (10% closed set + Zigbee)")
    print(f"  - Test data will be prepared separately with remaining 10% closed set + DSSS")
    
    return train_loader, threshold_loader

def prepare_test_loader(batch_size):
    """
    Prepares test loader with proper open set structure:
    - Known test samples: 10% of closed set (LTE, BT LE, WLAN) reserved from training
    - Unknown test samples: DSSS samples
    """
    print("\n--- Preparing Final Test DataLoader ---")
    
    # Load the closed set data again to get the reserved 10%
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1'])
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2'])
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("Main data files could not be loaded.")
    
    # Recreate the same closed set data structure
    snr_values_part1 = [5, 10, 15, 20, 25]
    num_samples_per_snr_part1 = data_part1.shape[1] // len(snr_values_part1)
    snr_labels_part1 = np.repeat(snr_values_part1, num_samples_per_snr_part1)

    snr_values_part2 = [-10, -5, 0]
    num_samples_per_snr_part2 = data_part2.shape[1] // len(snr_values_part2)
    snr_labels_part2 = np.repeat(snr_values_part2, num_samples_per_snr_part2)

    combined_data = np.concatenate((data_part1, data_part2), axis=1)
    combined_snrs = np.concatenate((snr_labels_part1, snr_labels_part2))
    
    # Apply same debug mode logic
    if config.DEBUG_MODE:
        p = np.random.permutation(combined_data.shape[1])
        subset_indices = p[:config.DEBUG_SUBSET_SIZE]
        combined_data = combined_data[:, subset_indices]
        combined_snrs = combined_snrs[subset_indices]
    
    # Extract the reserved 10% test samples from closed set
    signals_np = combined_data[:config.DATA_LEN, :]
    labels_int = np.argmax(np.abs(combined_data[config.DATA_LEN:, :]), axis=0).astype(int)
    
    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    snrs_tensor = torch.from_numpy(combined_snrs).to(torch.long)
    
    # Use the same reserved test indices
    test_closed_signals = signals[test_known_indices_global]
    test_closed_labels = labels[test_known_indices_global]
    test_closed_snrs = snrs_tensor[test_known_indices_global]
    
    # Load DSSS data for unknown testing
    dsss_signals, dsss_labels, dsss_snrs = prepare_dsss_test_data()
    
    if dsss_signals is not None:
        dsss_signals_tensor = torch.from_numpy(dsss_signals.T).to(torch.complex64)
        dsss_labels_tensor = torch.from_numpy(dsss_labels).to(torch.long)
        dsss_snrs_tensor = torch.from_numpy(dsss_snrs).to(torch.long)
        
        # Combine closed set test + DSSS test
        test_signals_final = torch.cat([test_closed_signals, dsss_signals_tensor])
        test_labels_final = torch.cat([test_closed_labels, dsss_labels_tensor])
        test_snrs_final = torch.cat([test_closed_snrs, dsss_snrs_tensor])
    else:
        # Fallback: use only closed set test data
        test_signals_final = test_closed_signals
        test_labels_final = test_closed_labels
        test_snrs_final = test_closed_snrs
    
    print(f"=== TEST DATA COMPOSITION ===")
    print(f"Closed set test samples (10%): {len(test_closed_signals)}")
    if dsss_signals is not None:
        print(f"DSSS test samples (unknown): {len(dsss_signals_tensor)}")
    print(f"Total test samples: {len(test_signals_final)}")
    unique_test_labels, counts = torch.unique(test_labels_final, return_counts=True)
    print(f"Test label distribution: {dict(zip(unique_test_labels.numpy(), counts.numpy()))}")
    print(f"=============================")
    
    test_dataset = SignalDataset(test_signals_final, test_labels_final, test_snrs_final)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Test Loader created with {len(test_loader.dataset)} samples.")
    
    return test_loader
