# config.py
import numpy as np

# --- 1. DATA AND CLASS CONFIGURATION ---
KNOWN_CLASSES_LIST = ['LTE', 'BT LE', 'WLAN']
KNOWN_UNKNOWN_CLASS = 'Zigbee'  # Used for thresholding
TEST_UNKNOWN_CLASS = 'DSSS'   # Used for final testing
DATA_LEN = 512
LABEL_LEN = 1

# --- DATA AND CLASS CONFIGURATION ---
# Paths for the main combined dataset (used for training and thresholding)
DATA_FILE_PATHS = {
    'path1': 'path/to/your/lte_bluetooth_le_wlan_mixeddata_withlabels_5dBto25dB_order.mat',
    'path2': 'path/to/your/lte_bluetooth_le_wlan_mixeddata_withlabels_-10dBto0dB_order.mat'
}

# --- NEW: Add paths for the separate test dataset ---
TEST_DATA_FILE_PATHS = {
    'known': 'path/to/your/test_known_signals.mat',    # e.g., valid_close_test
    'unknown': 'path/to/your/test_DSSS_signals.mat' # e.g., data_DSSS
}

# --- 2. MODEL ARCHITECTURE PARAMETERS ---
# These must match for training and evaluation
FEATURE_EXTRACTOR_PARAMS = {
    'num_classes': len(KNOWN_CLASSES_LIST),
    'feature_dim': 16
}

DIFFUSION_PARAMS = {
    'feature_dim': FEATURE_EXTRACTOR_PARAMS['feature_dim'],
    'sample_rate': DATA_LEN,
    'input_dim': 1,
    'hidden_dim': 128,
    'embed_dim': 128,
    'num_heads': 4,
    'num_block': 4,
    'max_step': 300,
    'noise_schedule': np.linspace(1e-4, 0.05, 300).tolist(),
}

# --- 3. TRAINING HYPERPARAMETERS ---
TRAINING_PARAMS = {
    'phase1_epochs': 10, # Epochs for Feature Extractor
    'phase2_epochs': 20, # Epochs for Diffusion Model
    'batch_size': 64,
    'lr_fe': 1e-3,       # Learning rate for Feature Extractor
    'lr_diff': 1e-4      # Learning rate for Diffusion Model
}

# --- 4. EVALUATION PARAMETERS ---
EVAL_PARAMS = {
    'batch_size': 64
}

# --- 5. FILE PATHS ---
PATHS = {
    'feature_extractor': 'disentangled_feature_extractor.pt',
    'diffusion_model': 'conditional_diffusion_model.pt'
}
