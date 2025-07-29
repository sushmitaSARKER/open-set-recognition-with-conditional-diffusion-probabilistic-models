# evaluate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Project-specific imports from our modular structure ---
# Make sure these files are in the same directory or accessible in your PYTHONPATH
import config
from data_loader import prepare_dataloaders
from models.feature_extractor import DisentangledFeatureExtractor
from models.diffusion_model import tfdiff_WiFi
from utils.diffusion_helper import SignalDiffusion
from engine import get_reconstruction_scores # We will use this from the engine

# ==============================================================================
# SECTION 1: THRESHOLD CALCULATION LOGIC
# ==============================================================================

def calculate_optimal_threshold(threshold_loader, feature_extractor, diffusion_model, diffusion_helper):
    """
    Calculates the optimal OSR threshold using a validation set and Youden's Index.
    
    Args:
        threshold_loader (DataLoader): DataLoader containing known and "known unknown" samples.
        feature_extractor (nn.Module): The trained feature extractor model.
        diffusion_model (nn.Module): The trained conditional diffusion model.
        diffusion_helper (SignalDiffusion): The diffusion process helper.
        
    Returns:
        float: The optimal threshold value for reconstruction error.
    """
    print("\n" + "="*50)
    print("### PHASE 3: Calculating Optimal Threshold ###")
    print("="*50)

    # 1. Get reconstruction scores for the entire thresholding dataset
    all_scores, all_labels = get_reconstruction_scores(
        loader=threshold_loader,
        feature_extractor=feature_extractor,
        diffusion_model=diffusion_model,
        diffusion_helper=diffusion_helper,
        params=config.DIFFUSION_PARAMS
    )
    
    # 2. Create binary labels (1 for known, 0 for unknown)
    true_binary_labels = np.array([1 if label < len(config.KNOWN_CLASSES_LIST) else 0 for label in all_labels])
    
    # 3. Calculate Youden's Index to find the best threshold
    # NOTE: roc_curve expects scores where higher means more likely to be in the positive class (known=1).
    # Since a LOW reconstruction error means "known", we pass the NEGATIVE error as the score.
    fpr, tpr, thresholds = roc_curve(true_binary_labels, -np.array(all_scores))
    
    j_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_score = thresholds[optimal_idx]
    
    # The threshold from roc_curve is based on the negative scores, so we flip it back
    optimal_threshold = -optimal_threshold_score

    print(f"\n--- Threshold Calculation Complete ---")
    print(f"Optimal Youden's Index: {j_scores[optimal_idx]:.4f}")
    print(f"Optimal Threshold (Reconstruction Error): {optimal_threshold:.6f}")
    
    return optimal_threshold

# ==============================================================================
# SECTION 2: FINAL EVALUATION LOGIC
# ==============================================================================

def run_final_evaluation(test_loader, optimal_threshold, feature_extractor, diffusion_model, diffusion_helper):
    """
    Runs a full evaluation on the test set using the calculated optimal threshold.
    Calculates detailed metrics, including per-SNR scores.
    """
    print("\n" + "="*50)
    print("### PHASE 4: Running Final Evaluation on Test Set ###")
    print("="*50)
    
    # 1. Get reconstruction scores for the entire test set
    all_scores, true_labels = get_reconstruction_scores(
        loader=test_loader,
        feature_extractor=feature_extractor,
        diffusion_model=diffusion_model,
        diffusion_helper=diffusion_helper,
        params=config.DIFFUSION_PARAMS
    )
    
    # 2. Apply threshold to get open-set predictions (known vs. unknown)
    # Prediction is "known" (1) if error is BELOW threshold, "unknown" (0) otherwise.
    open_set_preds_binary = (all_scores < optimal_threshold).astype(int)
    true_open_set_labels_binary = (true_labels < len(config.KNOWN_CLASSES_LIST)).astype(int)
    
    # 3. For samples predicted as "known", get the specific class prediction
    final_predictions = np.full_like(true_labels, -1) # Default to -1 for unknown
    
    # Find indices of samples that were classified as "known"
    known_indices = np.where(open_set_preds_binary == 1)[0]
    
    if len(known_indices) > 0:
        print(f"Out of {len(all_scores)} test samples, {len(known_indices)} were classified as 'known'. Getting their specific class...")
        # Get the signals for these specific indices to run through the classifier
        signals_to_classify = torch.stack([test_loader.dataset[i][0] for i in known_indices])
        
        with torch.no_grad():
            device = next(feature_extractor.parameters()).device
            signals_time = signals_to_classify.to(device)
            signals_freq = torch.fft.fft(signals_time)
            logits, _, _ = feature_extractor(signals_time, signals_freq)
            closed_set_preds = logits[2].argmax(dim=1).cpu().numpy()
            final_predictions[known_indices] = closed_set_preds
            
    # 4. Calculate and Print Metrics
    # Map true labels to the final format (-1 for all unknown classes)
    true_final_labels = true_labels.copy()
    true_final_labels[true_final_labels >= len(config.KNOWN_CLASSES_LIST)] = -1
    
    print("\n--- Overall Performance Metrics ---")
    accuracy = accuracy_score(true_final_labels, final_predictions)
    f1 = f1_score(true_final_labels, final_predictions, average='macro', zero_division=0)
    precision = precision_score(true_final_labels, final_predictions, average='macro', zero_division=0)
    recall = recall_score(true_final_labels, final_predictions, average='macro', zero_division=0)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")

    # Open vs. Closed Accuracy
    known_acc = accuracy_score(true_open_set_labels_binary[true_open_set_labels_binary==1], open_set_preds_binary[true_open_set_labels_binary==1])
    unknown_acc = accuracy_score(true_open_set_labels_binary[true_open_set_labels_binary==0], open_set_preds_binary[true_open_set_labels_binary==0])
    print(f"\nKnown Class Detection Accuracy (Sensitivity): {known_acc:.4f}")
    print(f"Unknown Class Detection Accuracy (Specificity): {unknown_acc:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix (Rows: True, Cols: Pred)")
    # We add -1 to the labels list to account for the 'Unknown' class
    cm_labels = list(range(len(config.KNOWN_CLASSES_LIST))) + [-1]
    cm = confusion_matrix(true_final_labels, final_predictions, labels=cm_labels)
    
    print("Labels:", [config.KNOWN_CLASSES_LIST[i] for i in range(len(config.KNOWN_CLASSES_LIST))] + ['Unknown'])
    print(cm)
    
    # NOTE: Add your per-SNR analysis here if needed. It would require the DataLoader
    # to also provide SNR information for each sample, which you can then use to
    # group the `all_scores` and `true_labels` arrays before calculating metrics.

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # This script assumes that the models have already been trained by train.py
    
    # 1. Setup environment and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # The loaders will contain all the necessary data splits
    _, threshold_loader, test_loader = prepare_dataloaders(config.EVAL_PARAMS['batch_size'])
    
    # 2. Load the trained models from Phase 1 and Phase 2
    print("Loading pre-trained models...")
    if not (os.path.exists(config.PATHS['feature_extractor']) and os.path.exists(config.PATHS['diffusion_model'])):
        print("\nERROR: Trained model files not found!")
        print("Please run the full training pipeline in 'train.py' first.")
    else:
        feature_extractor = DisentangledFeatureExtractor(
            num_classes=config.FEATURE_EXTRACTOR_PARAMS['num_classes'],
            feature_dim=config.FEATURE_EXTRACTOR_PARAMS['feature_dim']
        ).to(device)
        feature_extractor.load_state_dict(torch.load(config.PATHS['feature_extractor'], map_location=device))
        
        diffusion_model = tfdiff_WiFi(config.DIFFUSION_PARAMS).to(device)
        diffusion_model.load_state_dict(torch.load(config.PATHS['diffusion_model'], map_location=device))
        
        diffusion_helper = SignalDiffusion(config.DIFFUSION_PARAMS)
        
        # 3. Calculate the optimal threshold using the threshold_loader
        optimal_threshold = calculate_optimal_threshold(
            threshold_loader,
            feature_extractor,
            diffusion_model,
            diffusion_helper
        )
        
        # 4. Run the final, detailed evaluation on the test_loader
        run_final_evaluation(
            test_loader,
            optimal_threshold,
            feature_extractor,
            diffusion_model,
            diffusion_helper
        )