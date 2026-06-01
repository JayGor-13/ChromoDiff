import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import set_seed, load_config, setup_logger, setup_dirs
from .dataset import get_dataloader, reverse_complement_tensor
from .diffusion import AbsorbingStateScheduler
from .models.unet import GenoDiff1D

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def linear_warmup(step, warmup_steps, base_lr):
    """Scale LR linearly from 0 -> base_lr over warmup_steps."""
    return base_lr * min(1.0, step / max(warmup_steps, 1))

def reverse_complement_logits(logits: torch.Tensor) -> torch.Tensor:
    comp_map = [3, 2, 1, 0, 4]
    reversed_logits = torch.flip(logits, dims=[-1])
    return reversed_logits[:, comp_map, :]

def train_model(config_path: str):
    # 1. Load config and setup utils
    config = load_config(config_path)
    set_seed(config.get("SEED", 42))
    logger = setup_logger("ChromoDiff.Train")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint_dir = config.get("CHECKPOINT_DIR", "outputs/checkpoints")
    data_dir = config.get("DATA_DIR", "data/processed")
    setup_dirs(checkpoint_dir)
    
    # 2. Get data loader
    train_data_path = os.path.join(data_dir, "X_healthy.npy")
    if not os.path.exists(train_data_path):
        logger.error(f"Training data not found at {train_data_path}. Please run preprocessing first.")
        return
        
    logger.info(f"Loading data from {train_data_path}...")
    train_loader = get_dataloader(
        data_path=train_data_path,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=0
    )
    
    # 3. Initialize model and scheduler
    vocab_size = config.get("VOCAB_SIZE", 6)
    hidden_dim = config.get("HIDDEN_DIM", 256)
    model = GenoDiff1D(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        attention_dropout=config.get("ATTENTION_DROPOUT", 0.1),
    ).to(device)
    
    num_steps = config.get("T_STEPS", 1000)
    scheduler_diffusion = AbsorbingStateScheduler(
        num_steps=num_steps,
        beta_start=config.get("BETA_START", 1e-4),
        beta_end=config.get("BETA_END", 0.02),
        min_corruption_rate=config.get("MIN_CORRUPTION_RATE", 0.15)
    ).to(device)
    
    # 4. Optimizer and LR Scheduler setup
    optimizer = optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get("T_0", 25),
        T_mult=1,
        eta_min=config.get("ETA_MIN", 1e-6)
    )
    
    # Mixed precision setup
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    best_loss = float("inf")
    train_history = []
    
    warmup_epochs = config.get("WARMUP_EPOCHS", 2)
    warmup_steps = warmup_epochs * len(train_loader)
    global_step = 0
    epochs = config.get("EPOCHS", 50)
    
    logger.info("Starting Unsupervised Diffusion Training...")
    logger.info(f"  Model params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    logger.info(f"  LR warmup    : {warmup_epochs} epochs ({warmup_steps} steps)")
    logger.info(f"  LR restarts  : every {config.get('T_0', 25)} epochs")
    logger.info(f"  RC augment   : 50% per batch")
    logger.info(f"  Timestep samp: importance-weighted (squared)")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}")
        for batch_idx, x_start in enumerate(progress_bar):
            # Apply linear learning rate warmup
            if global_step < warmup_steps:
                warm_lr = linear_warmup(global_step, warmup_steps, config["LEARNING_RATE"])
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr
                    
            x_start = x_start.to(device, non_blocking=True)
            
            # Reverse complement augmentation (50% chance per batch)
            if torch.rand(1).item() < 0.5:
                x_start = reverse_complement_tensor(x_start)
            
            # Sample importance-weighted diffusion timesteps (biased toward low/mid t)
            t = scheduler_diffusion.sample_timesteps(x_start.shape[0], device)
            
            # Apply forward diffusion (adds [MASK] tokens)
            x_noisy, mutate_mask = scheduler_diffusion.q_sample(x_start, t)
            
            optimizer.zero_grad()
            
            # Run model with autocast
            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted_logits = model(x_noisy, t)
                
                # Masked cross entropy loss (vocabulary size is vocab_size - 1 = 5)
                mask_flat = mutate_mask.view(-1)
                logits_flat = predicted_logits.permute(0, 2, 1).reshape(-1, vocab_size - 1)
                labels_flat = x_start.view(-1)
                
                logits_masked = logits_flat[mask_flat]
                labels_masked = labels_flat[mask_flat]
                
                if mask_flat.sum() > 0:
                    loss_ce = F.cross_entropy(logits_masked, labels_masked)
                else:
                    loss_ce = F.cross_entropy(logits_flat, labels_flat)
                
                # Double-strand consistency loss
                x_noisy_rc = reverse_complement_tensor(x_noisy)
                predicted_logits_rc = model(x_noisy_rc, t)
                target_logits_rc = reverse_complement_logits(predicted_logits).detach()
                loss_dsc = F.mse_loss(predicted_logits_rc, target_logits_rc)
                
                loss = loss_ce + 0.1 * loss_dsc
            
            # Backpropagation using gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("GRAD_CLIP", 1.0))
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{get_lr(optimizer):.2e}"
            })
            
        # Step the learning rate scheduler (after warmup phase)
        if global_step >= warmup_steps:
            lr_scheduler.step(epoch - warmup_epochs + 1)
            
        avg_loss = epoch_loss / len(train_loader)
        train_history.append(avg_loss)
        
        logger.info(f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | LR: {get_lr(optimizer):.2e}")
        
        # Save epoch checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, f"genodiff_epoch_{epoch:03d}.pth"))
        
        # Keep best loss weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "genodiff_best.pth"))
            logger.info(f"  New best model saved with Loss: {best_loss:.4f}")
            
    logger.info(f"Training completed successfully! Best loss: {best_loss:.4f}")
    
    # Save loss history plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_history, lw=2, color="steelblue", label="Avg Masked Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ChromoDiff Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "training_curve.png"), dpi=150)
    plt.close()
    logger.info(f"Saved loss curve to {os.path.join(checkpoint_dir, 'training_curve.png')}")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/base_config.yaml"
    train_model(config_file)
