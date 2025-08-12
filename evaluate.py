import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Project-specific imports ---
import config
from data_loader import prepare_train_and_threshold_loaders, prepare_test_loader
from models.feature_extractor import DisentangledFeatureExtractor
from models.diffusion_model import tfdiff_WiFi
from utils.diffusion_helper import SignalDiffusion
from engine import get_reconstruction_scores, calculate_model_size

# ==============================================================================
# SECTION 1: THRESHOLD CALCULATION LOGIC
# ==============================================================================

def calculate_optimal_threshold(threshold_loader, feature_extractor, diffusion_model, diffusion_helper):
    """
    Calculates the optimal OSR threshold using a validation set and Youden's Index.
    """
    print("\n" + "="*50)
    print("### PHASE 3: Calculating Optimal Threshold ###")
    print("="*50)
    
    # Get reconstruction scores for the entire thresholding dataset
    all_scores, all_labels = get_reconstruction_scores(
    loader=threshold_loader,
    feature_extractor=feature_extractor,
    diffusion_model=diffusion_model,
    diffusion_helper=diffusion_helper,
    params=config.DIFFUSION_PARAMS
)

# Debug
unique, counts = np.unique(all_labels, return_counts=True)
print(f"Threshold labels distribution: {dict(zip(unique, counts))}")
print("Expected in threshold set: 0,1,2 (closed) and 3 (Zigbee)")

# Binary labels for ROC: known=1 if in {0,1,2}; unknown=0 if label==3
true_binary = np.array([1 if l in (0,1,2) else 0 for l in all_labels])

if len(np.unique(true_binary)) < 2:
    print("Only one class in threshold set. Using 95th-percentile fallback.")
    thr = np.percentile(all_scores, 95)
    print(f"Fallback threshold: {thr:.6f}")
    np.savetxt("optimal_threshold.txt", [thr])
    return thr

# Reconstruction error low => known; pass negative to roc_curve
fpr, tpr, thresholds = roc_curve(true_binary, -np.array(all_scores))
j = tpr - fpr
idx = np.argmax(j)
optimal_threshold = -thresholds[idx]

print("\n--- Threshold Calculation Complete ---")
print(f"Optimal Youden's Index: {j[idx]:.4f}")
print(f"Optimal Threshold (Reconstruction Error): {optimal_threshold:.6f}")

np.savetxt("optimal_threshold.txt", [optimal_threshold])
print("Saved optimal_threshold.txt")

return optimal_threshold

# ==============================================================================
# SECTION 2: FINAL EVALUATION LOGIC
# ==============================================================================

def run_final_evaluation(test_loader, optimal_threshold, feature_extractor, diffusion_model, diffusion_helper):
    """
    Runs a full evaluation on the test set using the calculated optimal threshold.
    """
    print("\n" + "="*50)
    print("### PHASE 4: Running Final Evaluation on Test Set ###")
    print("="*50)
    
    # get_reconstruction_scores now returns all_snrs as the third item

    all_scores, true_labels = get_reconstruction_scores(
    loader=test_loader,
    feature_extractor=feature_extractor,
    diffusion_model=diffusion_model,
    diffusion_helper=diffusion_helper,
    params=config.DIFFUSION_PARAMS
)

unique, counts = np.unique(true_labels, return_counts=True)
print(f"Test labels distribution: {dict(zip(unique, counts))}")
print("Class map: 0=LTE, 1=BT LE, 2=WLAN, 3=Zigbee, 4=DSSS")

# Open-set decision: 1=known if error<thr, else 0=unknown
open_known_pred = (all_scores < optimal_threshold).astype(int)

# We will produce a final 5-class prediction array:
# - For samples predicted known, we classify into 0/1/2 using feature_extractor
# - Zigbee (3) is considered known for open-set detection, but the classifier has num_classes=3
#   so we only produce 0/1/2 predictions. We'll leave Zigbee as 3 in ground truth for reporting.
final_preds = np.full_like(true_labels, -1)  # default unknown

# Indices predicted as known
known_idx = np.where(open_known_pred == 1)
if len(known_idx) > 0:
    print(f"{len(known_idx)}/{len(true_labels)} predicted as known -> classifying into 0/1/2...")
    signals_to_classify = torch.stack([test_loader.dataset[i] for i in known_idx])
    with torch.no_grad():
        device = next(feature_extractor.parameters()).device
        x_time = signals_to_classify.to(device)
        x_freq = torch.fft.fft(x_time)
        logits, _, _ = feature_extractor(x_time, x_freq)
        cls = logits.argmax(dim=1).cpu().numpy()  # values in {0,1,2}[2]
        final_preds[known_idx] = cls
else:
    print("No samples predicted as known by threshold.")

# -----------------------------
# CLOSED-SET METRICS (0,1,2 only)
# -----------------------------
print("\n" + "="*50)
print("CLOSED-SET EVALUATION (LTE, BT LE, WLAN)")
print("="*50)

closed_mask = (true_labels < 3)
if np.sum(closed_mask) > 0:
    y_true_closed = true_labels[closed_mask]
    y_pred_closed = final_preds[closed_mask]

    # Exclude samples that were predicted unknown (-1) from closed-set metrics
    valid = (y_pred_closed != -1)
    if np.any(valid):
        acc_closed = accuracy_score(y_true_closed[valid], y_pred_closed[valid])
        f1_closed = f1_score(y_true_closed[valid], y_pred_closed[valid], average='macro', zero_division=0)
        prec_closed = precision_score(y_true_closed[valid], y_pred_closed[valid], average='macro', zero_division=0)
        rec_closed = recall_score(y_true_closed[valid], y_pred_closed[valid], average='macro', zero_division=0)

        print(f"Closed-set Accuracy: {acc_closed:.4f}")
        print(f"Closed-set Macro F1: {f1_closed:.4f}")
        print(f"Closed-set Macro Precision: {prec_closed:.4f}")
        print(f"Closed-set Macro Recall: {rec_closed:.4f}")

        cm_closed = confusion_matrix(y_true_closed[valid], y_pred_closed[valid], labels=)[1][2]
        print("Closed-set Confusion Matrix [LTE, BT LE, WLAN]:")
        print(cm_closed)
    else:
        print("No valid closed-set predictions (all rejected by threshold).")
else:
    print("No closed-set samples present in test set.")

# -----------------------------
# OPEN-SET DETECTION METRICS
# -----------------------------
print("\n" + "="*50)
print("OPEN-SET DETECTION (Known vs Unknown)")
print("="*50)

# For detection: treat known={0,1,2,3}, unknown={4}
true_binary = np.array([1 if l in (0,1,2,3) else 0 for l in true_labels])  # 1=known, 0=unknown

sens = accuracy_score(true_binary[true_binary==1], open_known_pred[true_binary==1])  # sensitivity on known
spec = accuracy_score(true_binary[true_binary==0], open_known_pred[true_binary==0])  # specificity on unknown
acc_detect = accuracy_score(true_binary, open_known_pred)
f1_detect = f1_score(true_binary, open_known_pred, average='binary', zero_division=0)
prec_detect = precision_score(true_binary, open_known_pred, average='binary', zero_division=0)
rec_detect = recall_score(true_binary, open_known_pred, average='binary', zero_division=0)

print(f"Known Detection Accuracy (Sensitivity): {sens:.4f}")
print(f"Unknown Detection Accuracy (Specificity): {spec:.4f}")
print(f"Open-set Detection Accuracy: {acc_detect:.4f}")
print(f"Open-set Detection F1: {f1_detect:.4f}")
print(f"Open-set Detection Precision: {prec_detect:.4f}")
print(f"Open-set Detection Recall: {rec_detect:.4f}")

# -----------------------------
# 5-CLASS SUMMARY REPORT
# -----------------------------
print("\n" + "="*50)
print("5-CLASS SUMMARY (LTE, BT LE, WLAN, Zigbee, Unknown(DSSS))")
print("="*50)

# Map predictions: classifier only outputs 0/1/2; leave Zigbee/DSSS logic as is:
# True labels remain {0,1,2,3,4}. Preds are {0,1,2,-1}. We’ll keep -1 as Unknown for DSSS.
y_true_5 = true_labels.copy()
y_pred_5 = final_preds.copy()
# When true is Zigbee (3): 
#  - if threshold kept it (pred != -1) but classifier can only say 0/1/2, so it will likely count as misclass.
#  - this is expected; we still show the confusion.

cm_5 = confusion_matrix(y_true_5, y_pred_5, labels=[0,1,2,3,-1])
print("Labels order: [LTE(0), BT LE(1), WLAN(2), Zigbee(3), Unknown(-1/DSSS)]")
print(cm_5)

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # This script assumes that the models have already been trained by train.py
    
    # Setup environment and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # The loaders will contain all the necessary data splits
    _, threshold_loader = prepare_train_and_threshold_loaders(config.EVAL_PARAMS['batch_size'])
    test_loader = prepare_test_loader(config.EVAL_PARAMS['batch_size'])
    
    # Load the trained models from Phase 1 and Phase 2
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
        
        # Model size analysis
        print("\n--- Loaded Feature Extractor Info ---")
        calculate_model_size(feature_extractor)
        
        diffusion_model = tfdiff_WiFi(config.DIFFUSION_PARAMS).to(device)
        diffusion_model.load_state_dict(torch.load(config.PATHS['diffusion_model'], map_location=device))
        
        print("\n--- Loaded Diffusion Model Info ---")
        calculate_model_size(diffusion_model)
        print("----------------------------------\n")
        
        diffusion_helper = SignalDiffusion(config.DIFFUSION_PARAMS)
        
        # Calculate the optimal threshold using the threshold_loader
        optimal_threshold = calculate_optimal_threshold(
            threshold_loader,
            feature_extractor,
            diffusion_model,
            diffusion_helper
        )
        
        # Run the final, detailed evaluation on the test_loader
        run_final_evaluation(
            test_loader,
            optimal_threshold,
            feature_extractor,
            diffusion_model,
            diffusion_helper
        )


