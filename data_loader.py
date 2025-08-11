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

def prepare_all_data():
    """
    Load and prepare all 5 classes of data with proper label assignment
    """
    print("--- Loading All 5 Classes of Data ---")
    
    # Load closed set data (LTE, BT LE, WLAN)
    print("Loading closed set data (LTE, BT LE, WLAN)...")
    data_part1 = load_and_process_single_file(config.DATA_FILE_PATHS['path1'])
    data_part2 = load_and_process_single_file(config.DATA_FILE_PATHS['path2'])
    
    if data_part1 is None or data_part2 is None:
        raise FileNotFoundError("Closed set data files could not be loaded.")
    
    # Combine closed set data (no SNR processing)
    closed_set_data = np.concatenate((data_part1, data_part2), axis=1)
    closed_set_signals = closed_set_data[:config.DATA_LEN, :]
    closed_set_labels_int = np.argmax(np.abs(closed_set_data[config.DATA_LEN:, :]), axis=0).astype(int)
    
    print(f"Closed set loaded: {closed_set_data.shape[1]} samples")
    print(f"Closed set label distribution: {dict(zip(*np.unique(closed_set_labels_int, return_counts=True)))}")
    
    # Load Zigbee + DSSS data
    print("Loading Zigbee and DSSS data...")
    zigbee_dsss_file1 = config.TEST_DATA_FILE_PATHS['known']  # Contains Zigbee + DSSS
    zigbee_dsss_file2 = config.TEST_DATA_FILE_PATHS['unknown']  # Contains more DSSS
    
    zigbee_dsss_data1 = load_and_process_single_file(zigbee_dsss_file1)
    zigbee_dsss_data2 = load_and_process_single_file(zigbee_dsss_file2)
    
    if zigbee_dsss_data1 is None or zigbee_dsss_data2 is None:
        raise FileNotFoundError("Zigbee/DSSS data files could not be loaded.")
    
    # Extract Zigbee and DSSS samples with manual labeling
    zigbee_signals = zigbee_dsss_data1[:config.DATA_LEN, :3200]  # First 3200 are Zigbee
    dsss_signals_1 = zigbee_dsss_data1[:config.DATA_LEN, 3200:]  # Rest are DSSS
    dsss_signals_2 = zigbee_dsss_data2[:config.DATA_LEN, :]  # Additional DSSS
    
    # Combine all DSSS data
    dsss_signals = np.concatenate([dsss_signals_1, dsss_signals_2], axis=1)
    
    # Create proper labels for 5-class system
    zigbee_labels = np.full(zigbee_signals.shape[1], 3)  # Zigbee = class 3
    dsss_labels = np.full(dsss_signals.shape[1], 4)      # DSSS = class 4
    
    print(f"Zigbee samples: {zigbee_signals.shape[1]}")
    print(f"DSSS samples: {dsss_signals.shape[1]}")
    
    return {
        'closed_set_signals': closed_set_signals,
        'closed_set_labels': closed_set_labels_int,
        'zigbee_signals': zigbee_signals,
        'zigbee_labels': zigbee_labels,
        'dsss_signals': dsss_signals,
        'dsss_labels': dsss_labels
    }

def prepare_train_and_threshold_loaders(batch_size):
    """
    Prepare training and threshold loaders with the specified 80/10/10 split:
    - Training: 80% of closed set only
    - Threshold: 10% of closed set + first 3200 Zigbee samples
    """
    print("--- Preparing Training and Threshold Loaders (80/10/10 Split) ---")
    
    # Load all data
    all_data = prepare_all_data()
    
    # Convert closed set to tensors
    closed_signals_tensor = torch.from_numpy(all_data['closed_set_signals'].T).to(torch.complex64)
    closed_labels_tensor = torch.from_numpy(all_data['closed_set_labels']).to(torch.long)
    
    # Implement 80/10/10 split on closed set
    total_closed = len(closed_labels_tensor)
    indices = torch.randperm(total_closed)
    
    train_end = int(0.8 * total_closed)
    threshold_end = int(0.9 * total_closed)
    
    train_indices = indices[:train_end]
    threshold_indices = indices[train_end:threshold_end]
    test_indices = indices[threshold_end:]  # Reserved for testing
    
    # Store test indices globally for later use
    global reserved_test_indices
    reserved_test_indices = test_indices
    
    print(f"Closed set split: Training={len(train_indices)}, Threshold={len(threshold_indices)}, Test={len(test_indices)}")
    
    # Create training loader (80% of closed set only)
    train_signals = closed_signals_tensor[train_indices]
    train_labels = closed_labels_tensor[train_indices]
    train_dataset = SignalDataset(train_signals, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create threshold loader (10% closed set + first 3200 Zigbee)
    threshold_closed_signals = closed_signals_tensor[threshold_indices]
    threshold_closed_labels = closed_labels_tensor[threshold_indices]
    
    # Add first 3200 Zigbee samples for threshold calculation
    zigbee_signals_tensor = torch.from_numpy(all_data['zigbee_signals'].T).to(torch.complex64)
    zigbee_labels_tensor = torch.from_numpy(all_data['zigbee_labels']).to(torch.long)
    
    threshold_zigbee_signals = zigbee_signals_tensor[:3200]  # First 3200 Zigbee
    threshold_zigbee_labels = zigbee_labels_tensor[:3200]
    
    # Combine threshold data
    threshold_signals = torch.cat([threshold_closed_signals, threshold_zigbee_signals])
    threshold_labels = torch.cat([threshold_closed_labels, threshold_zigbee_labels])
    threshold_dataset = SignalDataset(threshold_signals, threshold_labels)
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training loader: {len(train_loader.dataset)} samples (closed set only)")
    print(f"Threshold loader: {len(threshold_loader.dataset)} samples (10% closed set + 3200 Zigbee)")
    
    return train_loader, threshold_loader

def prepare_test_loader(batch_size):
    """
    Prepare test loader with: 10% closed set + rest of Zigbee + all DSSS
    """
    print("--- Preparing Test Loader (10% closed set + rest of Zigbee + DSSS) ---")
    
    # Load all data
    all_data = prepare_all_data()
    
    # Get reserved 10% of closed set
    closed_signals_tensor = torch.from_numpy(all_data['closed_set_signals'].T).to(torch.complex64)
    closed_labels_tensor = torch.from_numpy(all_data['closed_set_labels']).to(torch.long)
    
    test_closed_signals = closed_signals_tensor[reserved_test_indices]
    test_closed_labels = closed_labels_tensor[reserved_test_indices]
    
    # Get rest of Zigbee samples (from 3201 onwards)
    zigbee_signals_tensor = torch.from_numpy(all_data['zigbee_signals'].T).to(torch.complex64)
    zigbee_labels_tensor = torch.from_numpy(all_data['zigbee_labels']).to(torch.long)
    
    if len(zigbee_signals_tensor) > 3200:
        test_zigbee_signals = zigbee_signals_tensor[3200:]  # Rest of Zigbee
        test_zigbee_labels = zigbee_labels_tensor[3200:]
    else:
        # If not enough Zigbee samples, use a different subset
        test_zigbee_signals = zigbee_signals_tensor[1600:3200]  # Alternative subset
        test_zigbee_labels = zigbee_labels_tensor[1600:3200]
    
    # Get all DSSS samples
    dsss_signals_tensor = torch.from_numpy(all_data['dsss_signals'].T).to(torch.complex64)
    dsss_labels_tensor = torch.from_numpy(all_data['dsss_labels']).to(torch.long)
    
    # Combine all test data
    test_signals = torch.cat([test_closed_signals, test_zigbee_signals, dsss_signals_tensor])
    test_labels = torch.cat([test_closed_labels, test_zigbee_labels, dsss_labels_tensor])
    
    # Create test dataset
    test_dataset = SignalDataset(test_signals, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"=== TEST DATA COMPOSITION ===")
    print(f"Closed set (10%): {len(test_closed_signals)} samples")
    print(f"Zigbee (rest): {len(test_zigbee_signals)} samples") 
    print(f"DSSS (all): {len(dsss_signals_tensor)} samples")
    print(f"Total test samples: {len(test_loader.dataset)}")
    
    # Show final label distribution
    unique_labels, counts = torch.unique(test_labels, return_counts=True)
    label_dist = dict(zip(unique_labels.numpy(), counts.numpy()))
    print(f"Test label distribution: {label_dist}")
    print(f"==============================")
    
    return test_loader

