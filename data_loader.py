# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from .config import *

class SignalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def prepare_dataloaders(batch_size=64):
    print("--- Preparing DataLoaders ---")
    if not (h5py and os.path.exists(DATA_FILE_PATH)):
        print(f"WARNING: Data file not found at '{DATA_FILE_PATH}' or h5py not installed.")
        print("Creating dummy data for demonstration purposes.")
        total_samples = 250000
        signals_np = np.random.randn(DATA_LEN, total_samples) + 1j * np.random.randn(DATA_LEN, total_samples)
        labels_int = np.array([0]*50000 + [1]*50000 + [2]*50000 + [3]*50000 + [4]*50000)
    else:
        print(f"Loading data from {DATA_FILE_PATH}...")
        with h5py.File(DATA_FILE_PATH, 'r') as f:
            data = f['vect'][:]
        signals_np = data[:DATA_LEN, :]
        labels_int = data[DATA_LEN, :].astype(int)

    signals = torch.from_numpy(signals_np.T).to(torch.complex64)
    labels = torch.from_numpy(labels_int).to(torch.long)
    
    known_indices = torch.cat([torch.where(labels == i)[0] for i in range(len(KNOWN_CLASSES_LIST))])
    known_unknown_indices = torch.where(labels == len(KNOWN_CLASSES_LIST))[0]
    test_unknown_indices = torch.where(labels == len(KNOWN_CLASSES_LIST) + 1)[0]
    
    train_dataset = SignalDataset(signals[known_indices], labels[known_indices])
    thresh_indices = torch.cat([known_indices, known_unknown_indices])
    threshold_dataset = SignalDataset(signals[thresh_indices], labels[thresh_indices])
    test_indices = torch.cat([known_indices, test_unknown_indices])
    test_dataset = SignalDataset(signals[test_indices], labels[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created.")
    return train_loader, threshold_loader, test_loader