import torch
import torch.nn as nn
import argparse
import os

# --- Project-specific imports ---
import config
from data_loader import prepare_train_and_threshold_loaders
from models.feature_extractor import DisentangledFeatureExtractor
from models.diffusion_model import tfdiff_WiFi
from utils.loss_functions import CosineSimilarityLoss
from utils.diffusion_helper import SignalDiffusion
import engine

# ==============================================================================
# PHASE 1: FEATURE EXTRACTOR TRAINING
# ==============================================================================
def run_phase1_training():
    """
    Orchestrates the training of the Disentangled Feature Extractor (Phase 1).
    """
    print("="*50)
    print("### STARTING PHASE 1: Training the Disentangled Feature Extractor ###")
    print("="*50)
    
    # Setup environment and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # We only need the training loader for this phase.
    # The dataloader will automatically be configured to only use the KNOWN classes.
     train_loader, _ = prepare_train_and_threshold_loaders(config.TRAINING_PARAMS['batch_size'])
    
    # Instantiate model, losses, and optimizer
    model = DisentangledFeatureExtractor(
        num_classes=config.FEATURE_EXTRACTOR_PARAMS['num_classes'],
        feature_dim=config.FEATURE_EXTRACTOR_PARAMS['feature_dim']
    ).to(device)
    
    loss_fns = {
        'ce': nn.CrossEntropyLoss(),
        'cos': CosineSimilarityLoss()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING_PARAMS['lr_fe'])
    
    # Main Training Loop
    for epoch in range(config.TRAINING_PARAMS['phase1_epochs']):
        # The core logic for one epoch is handled by the engine
        avg_loss, avg_acc = engine.train_fe_epoch(model, train_loader, optimizer, loss_fns, device)
        
        print(f"Epoch {epoch+1}/{config.TRAINING_PARAMS['phase1_epochs']} -> Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            engine.save_checkpoint(model, optimizer, epoch, avg_loss, f"checkpoint_fe_epoch_{epoch+1}.pt")
        
    # 4. Save the final trained model
    torch.save(model.state_dict(), config.PATHS['feature_extractor'])
    print(f"\n--- Phase 1 Complete. Feature extractor saved to '{config.PATHS['feature_extractor']}' ---")

# ==============================================================================
# PHASE 2: CONDITIONAL DIFFUSION MODEL TRAINING
# ==============================================================================
def run_phase2_training():
    """
    Orchestrates the training of the Conditional RF-Diffusion Model (Phase 2).
    """
    print("\n" + "="*50)
    print("### STARTING PHASE 2: Training the Conditional RF-Diffusion Model ###")
    print("="*50)
    
    # Setup environment and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, _ = prepare_train_and_threshold_loaders(config.TRAINING_PARAMS['batch_size'])
    
    # Load the pre-trained feature extractor from Phase 1
    print(f"Loading pre-trained feature extractor from '{config.PATHS['feature_extractor']}'...")
    if not os.path.exists(config.PATHS['feature_extractor']):
        print("\nERROR: Feature extractor model not found!")
        print("Please run Phase 1 training first by running: python train.py --phase 1")
        return
        
    feature_extractor = DisentangledFeatureExtractor(
        num_classes=config.FEATURE_EXTRACTOR_PARAMS['num_classes'],
        feature_dim=config.FEATURE_EXTRACTOR_PARAMS['feature_dim']
    ).to(device)
    feature_extractor.load_state_dict(torch.load(config.PATHS['feature_extractor'], map_location=device))
    feature_extractor.eval() # Set to evaluation mode; its weights are frozen
    
    # 3. Instantiate the diffusion model and its helpers
    diffusion_model = tfdiff_WiFi(config.DIFFUSION_PARAMS).to(device)
    diffusion_helper = SignalDiffusion(config.DIFFUSION_PARAMS)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=config.TRAINING_PARAMS['lr_diff'])
    
    # Main Training Loop
    for epoch in range(config.TRAINING_PARAMS['phase2_epochs']):
        avg_loss = engine.train_diffusion_epoch(
            diffusion_model=diffusion_model,
            feature_extractor=feature_extractor,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            diffusion_helper=diffusion_helper,
            params=config.DIFFUSION_PARAMS,
            device=device
        )
        print(f"Epoch {epoch+1}/{config.TRAINING_PARAMS['phase2_epochs']} -> Avg Diffusion Loss: {avg_loss:.6f}")
        
    # 5. Save the final trained model
    torch.save(diffusion_model.state_dict(), config.PATHS['diffusion_model'])
    print(f"\n--- Phase 2 Complete. Diffusion model saved to '{config.PATHS['diffusion_model']}' ---")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Train models for RF Open-Set Recognition.")
    parser.add_argument('--phase', type=int, choices=[1, 2], help="Which training phase to run (1 for Feature Extractor, 2 for Diffusion Model). Runs both if not specified.")
    args = parser.parse_args()

    if args.phase is None:
        run_phase1_training()
        run_phase2_training()
    elif args.phase == 1:
        run_phase1_training()
    elif args.phase == 2:
        run_phase2_training()

