import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ==============================================================================
# PHASE 1: FEATURE EXTRACTOR TRAINING LOGIC
# ==============================================================================

def train_fe_epoch(model, loader, optimizer, loss_fns, device):
    """
    Runs a single training epoch for the Disentangled Feature Extractor.
    
    Args:
        model (nn.Module): The DisentangledFeatureExtractor model.
        loader (DataLoader): The training DataLoader (known classes only).
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fns (dict): A dictionary containing 'ce' (CrossEntropy) and 'cos' (CosineSimilarity) losses.
        device (torch.device): The device to train on ('cuda' or 'cpu').
        
    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, total_acc, count = 0, 0, 0
    
    for signals, labels, snrs in tqdm(loader, desc="Training FE Epoch"):
        optimizer.zero_grad()
        signals, labels = signals.to(device), labels.to(device)
        
        x_time = signals
        x_freq = torch.fft.fft(signals)
        
        logits_list, features_list, _ = model(x_time, x_freq)
        
        loss_t = loss_fns['ce'](logits_list[0], labels)
        loss_f = loss_fns['ce'](logits_list[1], labels)
        loss_comb = loss_fns['ce'](logits_list[2], labels)
        loss_cos = loss_fns['cos'](features_list[0], features_list[1])
        
        # Total loss: We want to MINIMIZE cross-entropy and MAXIMIZE cosine similarity (difference)
        # Maximizing cosine similarity is the same as minimizing its negative.
        # However, the paper aims to make features ORTHOGONAL, meaning cosine similarity should be close to 0.
        # Therefore, we treat its absolute value as a penalty.
        loss = loss_t + loss_f + loss_comb + 0.5 * torch.abs(loss_cos)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += (logits_list[2].argmax(dim=1) == labels).sum().item()
        count += len(labels)
        
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / count
    return avg_loss, avg_acc

# ==============================================================================
# PHASE 2: CONDITIONAL DIFFUSION MODEL TRAINING LOGIC
# ==============================================================================

def train_diffusion_epoch(diffusion_model, feature_extractor, loader, optimizer, loss_fn, diffusion_helper, params, device):
    """
    Runs a single training epoch for the Conditional RF-Diffusion model.
    """
    diffusion_model.train()
    feature_extractor.eval() # Feature extractor is frozen
    total_loss = 0
    
    for signals, labels, snrs in tqdm(loader, desc="Training Diffusion Epoch"):
        optimizer.zero_grad()
        x_0 = signals.to(device)
        
        # Get the conditioning vector 'c' from the trained feature extractor
        with torch.no_grad():
            x_freq = torch.fft.fft(x_0)
            _, _, c = feature_extractor(x_0, x_freq)

        # Perform one step of diffusion training
        t = torch.randint(0, diffusion_helper.max_step, (x_0.shape[0],), device=device)
        x_0_reshaped = x_0.reshape(x_0.shape[0], params['sample_rate'], params['input_dim'])
        x_t = diffusion_helper.degrade_fn(x_0_reshaped, t)
        
        # Predict the original signal
        predicted_x_0 = diffusion_model(x_t, t, c)
        
        loss = loss_fn(predicted_x_0, x_0_reshaped)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    return avg_loss

# ==============================================================================
# PHASE 3 & 4: INFERENCE AND SCORE CALCULATION LOGIC
# ==============================================================================

def get_reconstruction_scores(loader, feature_extractor, diffusion_model, diffusion_helper, params):
    """
    Calculates reconstruction error (anomaly score) for all samples in a dataloader.
    Optimized for batch processing and memory efficiency.
    """
    device = next(diffusion_model.parameters()).device
    feature_extractor.eval()
    diffusion_model.eval()
    
    all_scores = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(loader, desc="Calculating Reconstruction Scores"):
            batch_size = signals.shape[0]
            signals = signals.to(device)
            
            # Process entire batch at once for feature extraction
            signals_freq = torch.fft.fft(signals)
            _, _, c = feature_extractor(signals, signals_freq)
            
            # Reshape signals for diffusion processing
            signals_reshaped = signals.reshape(batch_size, params['sample_rate'], params['input_dim'])
            
            # Batch denoising process
            t_max = torch.full((batch_size,), diffusion_helper.max_step - 1, device=device)
            x_t = diffusion_helper.degrade_fn(signals_reshaped, t_max)
            x_s = x_t
            
            # Reverse diffusion process (batch-wise)
            for s in range(diffusion_helper.max_step - 1, -1, -1):
                t_tensor = torch.full((batch_size,), s, device=device)
                predicted_x_0 = diffusion_model(x_s, t_tensor, c)
                
                if s > 0:
                    t_prev = torch.full((batch_size,), s - 1, device=device)
                    x_s = diffusion_helper.degrade_fn(predicted_x_0, t_prev)
                else:
                    x_s = predicted_x_0
            
            # Calculate reconstruction errors for the batch
            errors = F.mse_loss(signals_reshaped, x_s, reduction='none')
            errors = errors.view(batch_size, -1).mean(dim=1)  # Average over signal dimensions
            
            # Store results
            all_scores.extend(errors.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
    
    return np.array(all_scores), np.array(all_labels), np.array(all_snrs)


