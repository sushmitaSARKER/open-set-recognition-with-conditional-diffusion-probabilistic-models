import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ==============================================================================
# PHASE 1: FEATURE EXTRACTOR TRAINING LOGIC
# ==============================================================================

def train_fe_epoch(model, loader, optimizer, loss_fns, device, scheduler=None):
    """
    Enhanced training epoch for the Disentangled Feature Extractor.
    """
    model.train()
    total_loss, total_acc, count = 0, 0, 0
    
    # Use mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for signals, labels, snrs in tqdm(loader, desc="Training FE Epoch"):
        optimizer.zero_grad()
        signals, labels = signals.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        if scaler:
            with torch.cuda.amp.autocast():
                x_time = signals
                x_freq = torch.fft.fft(signals)
                logits_list, features_list, _ = model(x_time, x_freq)
                
                loss_t = loss_fns['ce'](logits_list[0], labels)
                loss_f = loss_fns['ce'](logits_list[1], labels)
                loss_comb = loss_fns['ce'](logits_list[2], labels)
                loss_cos = loss_fns['cos'](features_list[0], features_list[1])
                
                # Improved loss weighting
                loss = loss_t + loss_f + loss_comb + 0.5 * torch.abs(loss_cos)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x_time = signals
            x_freq = torch.fft.fft(signals)
            logits_list, features_list, _ = model(x_time, x_freq)
            
            loss_t = loss_fns['ce'](logits_list[0], labels)
            loss_f = loss_fns['ce'](logits_list[1], labels)
            loss_comb = loss_fns['ce'](logits_list[2], labels)
            loss_cos = loss_fns['cos'](features_list[0], features_list[1])
            
            loss = loss_t + loss_f + loss_comb + 0.5 * torch.abs(loss_cos)
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        total_acc += (logits_list[2].argmax(dim=1) == labels).sum().item()
        count += len(labels)
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / count
    
    return avg_loss, avg_acc


# ==============================================================================
# PHASE 2: CONDITIONAL DIFFUSION MODEL TRAINING LOGIC
# ==============================================================================

def train_diffusion_epoch(diffusion_model, feature_extractor, loader, optimizer, loss_fn, diffusion_helper, params, device, scheduler=None):
    """
    Enhanced training epoch for the Conditional RF-Diffusion model.
    """
    diffusion_model.train()
    feature_extractor.eval()
    total_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for signals, labels, snrs in tqdm(loader, desc="Training Diffusion Epoch"):
        optimizer.zero_grad()
        x_0 = signals.to(device, non_blocking=True)
        
        # Get the conditioning vector 'c' from the trained feature extractor
        with torch.no_grad():
            x_freq = torch.fft.fft(x_0)
            _, _, c = feature_extractor(x_0, x_freq)
        
        if scaler:
            with torch.cuda.amp.autocast():
                # Random timestep sampling
                t = torch.randint(0, diffusion_helper.max_step, (x_0.shape[0],), device=device)
                x_0_reshaped = x_0.reshape(x_0.shape[0], params['sample_rate'], params['input_dim'])
                x_t = diffusion_helper.degrade_fn(x_0_reshaped, t)
                
                predicted_x_0 = diffusion_model(x_t, t, c)
                loss = loss_fn(predicted_x_0, x_0_reshaped)
            
            scaler.scale(loss).backward()
            # Gradient clipping for stable training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            t = torch.randint(0, diffusion_helper.max_step, (x_0.shape[0],), device=device)
            x_0_reshaped = x_0.reshape(x_0.shape[0], params['sample_rate'], params['input_dim'])
            x_t = diffusion_helper.degrade_fn(x_0_reshaped, t)
            
            predicted_x_0 = diffusion_model(x_t, t, c)
            loss = loss_fn(predicted_x_0, x_0_reshaped)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
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

with torch.no_grad():
    for batch in tqdm(loader, desc="Calculating Reconstruction Scores"):
        signals, labels = batch

        signals = signals.to(device, non_blocking=True)
        labels_np = labels.cpu().numpy()

        B = signals.shape

        # Compute conditioning c for the whole batch
        signals_freq = torch.fft.fft(signals)
        _, _, c = feature_extractor(signals, signals_freq)

        # Reshape to [B, sample_rate, 1]
        x0 = signals.reshape(B, params['sample_rate'], params['input_dim'])

        # Degrade once to max step
        t_max = torch.full((B,), diffusion_helper.max_step - 1, device=device, dtype=torch.long)
        x_t = diffusion_helper.degrade_fn(x0, t_max)

        # Reverse diffusion, batched
        x_s = x_t
        for s in range(diffusion_helper.max_step - 1, -1, -1):
            t_tensor = torch.full((B,), s, device=device, dtype=torch.long)
            x0_hat = diffusion_model(x_s, t_tensor, c)
            if s > 0:
                t_prev = torch.full((B,), s - 1, device=device, dtype=torch.long)
                x_s = diffusion_helper.degrade_fn(x0_hat, t_prev)
            else:
                x_s = x0_hat

        # Reconstruction error per sample
        # x0, x_s both [B, T, C]; compute MSE over (T,C), reduce='none' then mean over dims
        errors = F.mse_loss(x0, x_s, reduction='none')  # [B, T, C]
        errors = err.view(B, -1).mean(dim=1)            # [B]

        # Store results
        all_scores.extend(err.detach().cpu().numpy())
        all_labels.extend(labels_np)

return np.array(all_scores), np.array(all_labels)

# ==============================================================================
# --------------------UTILITY FUNCTIONS ------------------
# ==============================================================================

def validate_model(model, loader, loss_fns, device):
    """
    Validation function for feature extractor.
    """
    model.eval()
    total_loss, total_acc, count = 0, 0, 0
    
    with torch.no_grad():
        for signals, labels, snrs in tqdm(loader, desc="Validation"):
            signals, labels = signals.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            x_time = signals
            x_freq = torch.fft.fft(signals)
            logits_list, features_list, _ = model(x_time, x_freq)
            
            loss_t = loss_fns['ce'](logits_list[0], labels)
            loss_f = loss_fns['ce'](logits_list[1], labels)
            loss_comb = loss_fns['ce'](logits_list[2], labels)
            loss_cos = loss_fns['cos'](features_list[0], features_list[1])
            
            loss = loss_t + loss_f + loss_comb + 0.5 * torch.abs(loss_cos)
            
            total_loss += loss.item()
            total_acc += (logits_list[2].argmax(dim=1) == labels).sum().item()
            count += len(labels)
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / count
    
    return avg_loss, avg_acc

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
    return model, optimizer, epoch, loss

def calculate_model_size(model):
    """
    Calculate and print model size information.
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {param_count:,}")
    print(f"Trainable parameters: {trainable_param_count:,}")
    print(f"Model size: {param_count * 4 / 1024 / 1024:.2f} MB (assuming 4 bytes per parameter)")
    
    return param_count, trainable_param_count






