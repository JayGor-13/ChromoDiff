import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from .utils import load_config, setup_logger, setup_dirs
from .models.unet import GenoDiff1D

def calculate_gves(model, seq_corrupted, ref_base, alt_base, mutation_pos=512, epsilon=1e-8):
    """
    Calculate GVES score for a single variant centered at mutation_pos.
    GVES = log(P_ref + eps) - log(P_alt + eps)
    """
    model.eval()
    device = next(model.parameters()).device
    
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    ref_idx = nuc_to_idx[ref_base] if isinstance(ref_base, str) else ref_base
    alt_idx = nuc_to_idx[alt_base] if isinstance(alt_base, str) else alt_base
    
    if seq_corrupted.dim() == 1:
        seq_corrupted = seq_corrupted.unsqueeze(0)
    
    seq_corrupted = seq_corrupted.to(device)
    B = seq_corrupted.shape[0]
    
    # Replace position 512 with MASK token (5)
    seq_masked = seq_corrupted.clone()
    seq_masked[:, mutation_pos] = 5
    
    t_tensor = torch.zeros(B, device=device, dtype=torch.long)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(seq_masked, t_tensor) # [B, 6, L]
        
        # Cast to float32 to avoid float16 underflow/overflow in log/softmax
        logits_f32 = logits.float()
        
        # Softmax over logits at mutation_pos
        probs = torch.softmax(logits_f32[:, :, mutation_pos], dim=1) # [B, 6]
        
        # Extract ref and alt probabilities
        p_ref = probs[:, ref_idx]
        p_alt = probs[:, alt_idx]
        
        gves = torch.log(p_ref + epsilon) - torch.log(p_alt + epsilon)
        
    return gves.cpu().numpy()

def score_dataset_gves(model, X_healthy, X_corrupted, mutation_pos=512, batch_size=64, epsilon=1e-8):
    """
    Score the dataset using the GVES score for each sequence.
    GVES = log(P_ref + eps) - log(P_alt + eps)
    """
    model.eval()
    device = next(model.parameters()).device
    N = len(X_healthy)
    
    all_gves = []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_h = X_healthy[start:end].to(device)
        batch_c = X_corrupted[start:end].to(device)
        B = batch_h.shape[0]
        
        # Replace position 512 with MASK token (5)
        batch_masked = batch_c.clone()
        batch_masked[:, mutation_pos] = 5
        
        t_tensor = torch.zeros(B, device=device, dtype=torch.long)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(batch_masked, t_tensor) # [B, 6, L]
            
            # Cast to float32 to avoid float16 underflow/overflow in log/softmax
            logits_f32 = logits.float()
            probs = torch.softmax(logits_f32[:, :, mutation_pos], dim=1) # [B, 6]
            
            # Extract ref and alt indices
            ref_idx = batch_h[:, mutation_pos] # [B]
            alt_idx = batch_c[:, mutation_pos] # [B]
            
            p_ref = probs[torch.arange(B), ref_idx]
            p_alt = probs[torch.arange(B), alt_idx]
            
            gves = torch.log(p_ref + epsilon) - torch.log(p_alt + epsilon)
            
        all_gves.append(gves.cpu().numpy())
        
    return np.concatenate(all_gves)

@torch.no_grad()
def score_dataset_percentile_nll(model, sequences, batch_size=64, timesteps=None, percentile=99.0):
    """
    Alternative evaluation strategy using NLL percentile and Z-score (from notebook).
    """
    model.eval()
    device = next(model.parameters()).device
    N = len(sequences)
    
    if timesteps is None:
        timesteps = [1, 2, 3, 5, 8]
        
    all_pct, all_z = [] , []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = sequences[start:end].to(device)
        B = batch.shape[0]
        SEQ_LEN = batch.shape[1]
        
        pct_acc = torch.zeros(B, SEQ_LEN, device=device)
        
        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(batch, t) # [B, 6, 1024]
                
            nll = F.cross_entropy(
                logits.permute(0,2,1).reshape(-1, 6).float(),
                batch.reshape(-1),
                reduction="none"
            ).reshape(B, SEQ_LEN)
            
            pct_acc += nll
            
        mean_nll_per_pos = (pct_acc / len(timesteps)).cpu().float()
        
        # 99th percentile NLL score
        pct_scores = torch.quantile(mean_nll_per_pos, percentile / 100.0, dim=1)
        
        # Z-score outlier detection
        seq_mean = mean_nll_per_pos.mean(dim=1, keepdim=True)
        seq_std = mean_nll_per_pos.std(dim=1, keepdim=True).clamp(min=1e-6)
        z_scores = ((mean_nll_per_pos - seq_mean) / seq_std).max(dim=1).values
        
        all_pct.append(pct_scores.numpy())
        all_z.append(z_scores.numpy())
        
    return np.concatenate(all_pct), np.concatenate(all_z)

def evaluate_predictions(y_true, scores, best_name="GVES", checkpoint_dir="outputs/checkpoints"):
    # Clean scores of NaN and Inf, and cast to float32
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    
    # Check if scores are inverted (e.g. higher score means more benign)
    auroc_flip = roc_auc_score(y_true, -scores)
    if auroc_flip > auroc:
        scores = -scores
        auroc = auroc_flip
        auprc = average_precision_score(y_true, scores)
        flip_note = " (flipped)"
    else:
        flip_note = ""
        
    # Save statistics plots
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    rand_auprc = y_true.mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"ChromoDiff Evaluation Metrics — {best_name} Scoring{flip_note}", fontsize=13, fontweight="bold")
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUROC={auroc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    axes[1].plot(rec, prec, color="coral", lw=2, label=f"AUPRC={auprc:.4f}")
    axes[1].axhline(rand_auprc, color="k", ls="--", lw=1, alpha=0.4, label=f"Random={rand_auprc:.3f}")
    axes[1].fill_between(rec, prec, alpha=0.1, color="coral")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Class Score Distribution
    axes[2].hist(scores[y_true == 0], bins=60, alpha=0.6, color="steelblue", label="Benign", density=True)
    axes[2].hist(scores[y_true == 1], bins=60, alpha=0.6, color="coral", label="Pathogenic", density=True)
    axes[2].set(xlabel="Score", ylabel="Density", title="Score Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "evaluation_metrics.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return auroc, auprc, plot_path

def run_evaluation(config_path: str, weights_path: str):
    config = load_config(config_path)
    logger = setup_logger("ChromoDiff.Evaluate")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint_dir = config.get("CHECKPOINT_DIR", "outputs/checkpoints")
    data_dir = config.get("DATA_DIR", "data/processed")
    setup_dirs(checkpoint_dir)
    
    # 1. Load clinical test sets
    logger.info("Loading validation datasets...")
    healthy_path = os.path.join(data_dir, "X_healthy.npy")
    corrupted_path = os.path.join(data_dir, "X_corrupted.npy")
    labels_path = os.path.join(data_dir, "Y_labels.npy")
    
    if not (os.path.exists(healthy_path) and os.path.exists(corrupted_path) and os.path.exists(labels_path)):
        logger.error("Dataset arrays not found. Please run preprocessing first.")
        return
        
    X_healthy = torch.tensor(np.load(healthy_path), dtype=torch.long)
    X_corrupted = torch.tensor(np.load(corrupted_path), dtype=torch.long)
    Y_labels = np.load(labels_path)
    
    # 2. Instantiate and load model
    vocab_size = config.get("VOCAB_SIZE", 6)
    hidden_dim = config.get("HIDDEN_DIM", 256)
    model = GenoDiff1D(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    
    logger.info(f"Loading pretrained weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    # Handle if state dict contains checkpoint metadata or raw weights
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
        
    # 3. Compute zero-shot pathogenicity scores using GVES
    logger.info("Computing zero-shot GVES pathogenicity scores...")
    gves_scores = score_dataset_gves(
        model=model,
        X_healthy=X_healthy,
        X_corrupted=X_corrupted,
        mutation_pos=512,
        batch_size=config.get("BATCH_SIZE", 64)
    )
    
    # 4. Report statistics
    auroc, auprc, plot_path = evaluate_predictions(
        y_true=Y_labels,
        scores=gves_scores,
        best_name="GVES",
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info("==================================================")
    logger.info("  Zero-Shot Variant Pathogenicity Metrics")
    logger.info("==================================================")
    logger.info(f"  AUROC : {auroc:.4f}")
    logger.info(f"  AUPRC : {auprc:.4f}")
    logger.info(f"  Saved evaluation figures → {plot_path}")
    logger.info("==================================================")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/base_config.yaml"
    weights_file = sys.argv[2] if len(sys.argv) > 2 else "outputs/checkpoints/genodiff_best.pth"
    run_evaluation(config_file, weights_file)
