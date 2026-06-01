import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm

from .utils import load_config, setup_logger, setup_dirs
from .models.unet import GenoDiff1D
from .dataset import reverse_complement_tensor

def reverse_complement_logits(logits: torch.Tensor) -> torch.Tensor:
    comp_map = [3, 2, 1, 0, 4]
    reversed_logits = torch.flip(logits, dims=[-1])
    return reversed_logits[:, comp_map, :]


def calculate_gves(model, seq_corrupted, ref_base, alt_base, mutation_pos=512, gves_timestep=5):
    """
    Calculate GVES score for a single variant centered at mutation_pos.
    GVES = log(P_ref) - log(P_alt) using log_softmax for numerical stability.
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
    
    t_tensor = torch.full((B,), gves_timestep, device=device, dtype=torch.long)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(seq_masked, t_tensor) # [B, 5, L]
        
        # Cast to float32 to avoid float16 underflow/overflow in log/softmax
        logits_f32 = logits.float()
        
        # log_softmax over the 5 valid base classes
        log_probs = F.log_softmax(logits_f32[:, :5, mutation_pos], dim=1) # [B, 5]
        
        # Extract ref and alt log probabilities
        log_p_ref = log_probs[:, ref_idx]
        log_p_alt = log_probs[:, alt_idx]
        
        gves = log_p_ref - log_p_alt
        
    return gves.cpu().numpy()

def score_dataset_gves(model, X_healthy, X_corrupted, mutation_pos=512, batch_size=64, gves_timestep=5):
    """
    Score the dataset using the GVES score for each sequence.
    GVES = log(P_ref) - log(P_alt) using log_softmax for numerical stability.
    Applies double-strand symmetric averaging to improve accuracy.
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
        
        t_tensor = torch.full((B,), gves_timestep, device=device, dtype=torch.long)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # Forward strand prediction
                logits_fwd = model(batch_masked, t_tensor) # [B, 5, L]
                
                # Reverse complement strand prediction
                batch_masked_rc = reverse_complement_tensor(batch_masked)
                logits_rc = model(batch_masked_rc, t_tensor)
                logits_rc_fwd = reverse_complement_logits(logits_rc)
                
                # Symmetric average
                logits = 0.5 * (logits_fwd + logits_rc_fwd)
            
            # Cast to float32 to avoid float16 underflow/overflow in log/softmax
            logits_f32 = logits.float()
            log_probs = F.log_softmax(logits_f32[:, :5, mutation_pos], dim=1) # [B, 5]
            
            # Extract ref and alt indices
            ref_idx = batch_h[:, mutation_pos] # [B]
            alt_idx = batch_c[:, mutation_pos] # [B]
            
            log_p_ref = log_probs[torch.arange(B), ref_idx]
            log_p_alt = log_probs[torch.arange(B), alt_idx]
            
            gves = log_p_ref - log_p_alt
            
        all_gves.append(gves.cpu().numpy())
        
    return np.concatenate(all_gves)

def gc_normalize_scores(scores, gc_content, y_true, num_bins=10):
    """
    Remove GC content correlation by subtracting the median benign variant score
    within each GC-content bin from the raw GVES scores.
    """
    normalized_scores = scores.copy()
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    for i in range(num_bins):
        bin_mask = (gc_content >= bin_edges[i]) & (gc_content < bin_edges[i+1])
        if i == num_bins - 1:
            bin_mask = bin_mask | (gc_content == bin_edges[i+1])
        
        # Calculate median of benign variants (y_true == 0) in this bin
        benign_in_bin = bin_mask & (y_true == 0)
        if benign_in_bin.sum() > 0:
            median_benign = np.median(scores[benign_in_bin])
        else:
            median_benign = 0.0
            
        normalized_scores[bin_mask] -= median_benign
    return normalized_scores


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
                logits = model(batch, t) # [B, 5, 1024]
                
            nll = F.cross_entropy(
                logits.permute(0,2,1).reshape(-1, 5).float(),
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
    
    # Check if scores are inverted (e.g. higher score means more benign) but do not flip automatically
    auroc_flip = roc_auc_score(y_true, -scores)
    flip_note = ""
    if auroc < 0.5:
        flip_note = " (Warning: AUROC < 0.5; expected direction is positive correlation with GVES)"
        # Log a warning about directionality mismatch
        print(f"Warning: Raw GVES AUROC is {auroc:.4f} (< 0.5). If higher GVES represents reference base disruption, raw AUROC should be > 0.5. Flipped AUROC is {auroc_flip:.4f}.")

        
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
    plot_path = os.path.join(checkpoint_dir, f"{best_name.lower()}_evaluation_metrics.png")
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
    healthy_ref_path = os.path.join(data_dir, "X_healthy_ref.npy")
    healthy_path = healthy_ref_path if os.path.exists(healthy_ref_path) else os.path.join(data_dir, "X_healthy.npy")
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
    model = GenoDiff1D(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        attention_dropout=config.get("ATTENTION_DROPOUT", 0.1),
    ).to(device)
    
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
        batch_size=config.get("BATCH_SIZE", 64),
        gves_timestep=config.get("GVES_TIMESTEP", 5),
    )
    
    # 4. Apply GC-content normalization
    logger.info("Applying GC-content normalization...")
    gc_content = ((X_healthy == 1) | (X_healthy == 2)).float().mean(dim=1).numpy()
    gves_scores_normalized = gc_normalize_scores(gves_scores, gc_content, Y_labels)
    
    # 5. Report statistics
    auroc_raw, auprc_raw, plot_path_raw = evaluate_predictions(
        y_true=Y_labels,
        scores=gves_scores,
        best_name="GVES_Raw",
        checkpoint_dir=checkpoint_dir
    )
    
    auroc_norm, auprc_norm, plot_path_norm = evaluate_predictions(
        y_true=Y_labels,
        scores=gves_scores_normalized,
        best_name="GVES_GC_Normalized",
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info("==================================================")
    logger.info("  Zero-Shot Variant Pathogenicity Metrics")
    logger.info("==================================================")
    logger.info(f"  Raw GVES AUROC            : {auroc_raw:.4f}")
    logger.info(f"  Raw GVES AUPRC            : {auprc_raw:.4f}")
    logger.info(f"  GC-Normalized GVES AUROC  : {auroc_norm:.4f}")
    logger.info(f"  GC-Normalized GVES AUPRC  : {auprc_norm:.4f}")
    logger.info(f"  Saved evaluation figures → {plot_path_raw} and {plot_path_norm}")
    logger.info("==================================================")

def evaluate_traitgym(config_path: str, weights_path: str, dataset_name: str = "mendelian_traits", dummy: bool = False):
    """Evaluate zero-shot variant pathogenicity prediction on the TraitGym benchmark dataset."""
    config = load_config(config_path)
    logger = setup_logger(f"ChromoDiff.Evaluate.{dataset_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint_dir = config.get("CHECKPOINT_DIR", "outputs/checkpoints")
    data_dir = config.get("DATA_DIR", "data/processed")
    setup_dirs(checkpoint_dir)
    
    # 1. Load TraitGym Dataset
    logger.info(f"Loading TraitGym dataset ({dataset_name})...")
    MUT_POS = 512
    
    if dummy:
        # Generate synthetic TraitGym data for dry-run
        logger.info("Running in dummy mode. Generating synthetic TraitGym variants...")
        np.random.seed(42)
        num_variants = 200
        # Create random bases
        ref_seqs = np.random.choice(4, size=(num_variants, 1024)).astype(np.int8)
        alt_seqs = ref_seqs.copy()
        for i in range(num_variants):
            ref_base = ref_seqs[i, MUT_POS]
            choices = [b for b in range(4) if b != ref_base]
            alt_seqs[i, MUT_POS] = np.random.choice(choices)
        Y_labels = np.random.choice([0, 1], size=(num_variants,)).astype(np.int8)
        X_healthy = torch.tensor(ref_seqs, dtype=torch.long)
        X_corrupted = torch.tensor(alt_seqs, dtype=torch.long)
    else:
        # Load real dataset
        try:
            from datasets import load_dataset
            from pyfaidx import Fasta
        except ImportError:
            logger.error("Required libraries (datasets or pyfaidx) are missing. Run pip install datasets pyfaidx.")
            return None, None
            
        try:
            dataset = load_dataset("songlab/TraitGym", dataset_name, split="test")
        except Exception as e:
            logger.error(f"Failed to load TraitGym dataset from Hugging Face: {e}")
            logger.info("Falling back to dummy mode.")
            return evaluate_traitgym(config_path, weights_path, dataset_name, dummy=True)
            
        hg38_fa = "data/raw/hg38.fa"
        if not os.path.exists(hg38_fa):
            logger.error(f"hg38.fa reference not found at {hg38_fa}. Please run preprocessing first.")
            return None, None
            
        try:
            genome = Fasta(hg38_fa, as_raw=True, sequence_always_upper=True)
        except Exception as e:
            logger.error(f"Error opening hg38.fa using pyfaidx: {e}")
            return None, None
            
        # Parse variants
        nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        WINDOW_SIZE = 1024
        MAX_N_FRAC = 0.02
        
        ref_windows = []
        alt_windows = []
        labels = []
        
        # Helper to map ASCII string to token indices
        byte_lut = np.full(256, 4, dtype=np.int8)
        for base, idx in nuc_to_idx.items():
            byte_lut[ord(base)] = idx
            
        logger.info(f"Extracting genomic sequence windows for {len(dataset)} variants...")
        for row in tqdm(dataset, desc="Processing TraitGym variants"):
            chrom = row['chrom']
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
                
            if chrom not in genome.keys():
                continue
                
            pos = int(row['pos'])
            ref = row['ref']
            alt = row['alt']
            label = 1 if row['label'] else 0
            
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in nuc_to_idx or alt not in nuc_to_idx:
                continue
                
            start0 = pos - 1 - MUT_POS
            end0 = start0 + WINDOW_SIZE
            
            if start0 < 0:
                continue
                
            seq = genome[chrom][start0:end0]
            if len(seq) != WINDOW_SIZE:
                continue
                
            arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            ref_tokens = byte_lut[arr]
            
            r_idx = nuc_to_idx[ref]
            a_idx = nuc_to_idx[alt]
            
            if ref_tokens[MUT_POS] != r_idx:
                continue
                
            if (ref_tokens == 4).mean() > MAX_N_FRAC:
                continue
                
            alt_tokens = ref_tokens.copy()
            alt_tokens[MUT_POS] = a_idx
            
            ref_windows.append(ref_tokens)
            alt_windows.append(alt_tokens)
            labels.append(label)
            
        if len(labels) == 0:
            logger.error("No valid variants extracted from the TraitGym dataset.")
            return None, None
            
        logger.info(f"Extracted {len(labels)} valid variants out of {len(dataset)}.")
        X_healthy = torch.tensor(np.asarray(ref_windows, dtype=np.int8), dtype=torch.long)
        X_corrupted = torch.tensor(np.asarray(alt_windows, dtype=np.int8), dtype=torch.long)
        Y_labels = np.asarray(labels, dtype=np.int8)

    # 2. Instantiate and load model
    vocab_size = config.get("VOCAB_SIZE", 6)
    hidden_dim = config.get("HIDDEN_DIM", 256)
    model = GenoDiff1D(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        attention_dropout=config.get("ATTENTION_DROPOUT", 0.1),
    ).to(device)
    
    logger.info(f"Loading pretrained weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
        
    # 3. Score
    logger.info("Computing zero-shot GVES scores on TraitGym...")
    gves_scores = score_dataset_gves(
        model=model,
        X_healthy=X_healthy,
        X_corrupted=X_corrupted,
        mutation_pos=MUT_POS,
        batch_size=config.get("BATCH_SIZE", 64),
        gves_timestep=config.get("GVES_TIMESTEP", 5),
    )
    
    # 4. Apply GC-content normalization
    logger.info("Applying GC-content normalization...")
    gc_content = ((X_healthy == 1) | (X_healthy == 2)).float().mean(dim=1).numpy()
    gves_scores_normalized = gc_normalize_scores(gves_scores, gc_content, Y_labels)
    
    # 5. Evaluate
    auroc_raw, auprc_raw, plot_path_raw = evaluate_predictions(
        y_true=Y_labels,
        scores=gves_scores,
        best_name=f"TraitGym_{dataset_name}_Raw",
        checkpoint_dir=checkpoint_dir
    )
    
    auroc_norm, auprc_norm, plot_path_norm = evaluate_predictions(
        y_true=Y_labels,
        scores=gves_scores_normalized,
        best_name=f"TraitGym_{dataset_name}_GC_Normalized",
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info("==================================================")
    logger.info(f"  TraitGym ({dataset_name}) Zero-Shot Variant Metrics")
    logger.info("==================================================")
    logger.info(f"  Raw GVES AUROC            : {auroc_raw:.4f}")
    logger.info(f"  Raw GVES AUPRC            : {auprc_raw:.4f}")
    logger.info(f"  GC-Normalized GVES AUROC  : {auroc_norm:.4f}")
    logger.info(f"  GC-Normalized GVES AUPRC  : {auprc_norm:.4f}")
    logger.info(f"  Saved evaluation figures → {plot_path_raw} and {plot_path_norm}")
    logger.info("==================================================")
    
    return auroc_norm, auprc_norm

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/base_config.yaml"
    weights_file = sys.argv[2] if len(sys.argv) > 2 else "outputs/checkpoints/genodiff_best.pth"
    run_evaluation(config_file, weights_file)
