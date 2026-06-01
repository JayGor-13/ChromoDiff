import os
import urllib.request
import gzip
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import load_config, setup_logger, setup_dirs

# Categorical mappings
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
IDX_TO_NUC = {v: k for k, v in NUC_TO_IDX.items()}

# Lookup table for fast ASCII mapping
BYTE_LUT = np.full(256, 4, dtype=np.int8)
for base, idx in NUC_TO_IDX.items():
    BYTE_LUT[ord(base)] = idx

CODING_CONSEQUENCES = (
    "missense_variant",
    "nonsense",
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "synonymous_variant",
    "coding_sequence_variant",
    "protein_altering_variant",
    "inframe_insertion",
    "inframe_deletion",
    "splice_acceptor_variant",
    "splice_donor_variant",
)

def seq_to_tokens(seq: str) -> np.ndarray:
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return BYTE_LUT[arr]

def download_file(url: str, dest_path: str, logger):
    """Download a file with progress updates."""
    if os.path.exists(dest_path):
        logger.info(f"File {dest_path} already exists. Skipping download.")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    
    # Custom block-wise download with tqdm progress bar
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)

def extract_gzip(src_path: str, dest_path: str, logger):
    """Extract gzip file."""
    if os.path.exists(dest_path):
        logger.info(f"Extracted file {dest_path} already exists. Skipping extraction.")
        return
        
    logger.info(f"Extracting {src_path} to {dest_path}...")
    with gzip.open(src_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    logger.info("Extraction complete.")

def generate_dummy_data(dest_dir: str, num_healthy: int = 20000, num_variant: int = 1000, seq_len: int = 1024, seed: int = 42):
    """Generate high-quality synthetic genomic datasets for fast dry-runs and pipeline verification."""
    np.random.seed(seed)
    setup_dirs(dest_dir)
    
    # Healthy genomic sequences (classes 0..3 representing A,C,G,T and occasional N=4)
    probs = [0.24, 0.24, 0.24, 0.24, 0.04]
    X_healthy = np.random.choice(5, size=(num_healthy, seq_len), p=probs).astype(np.int8)
    
    # Variant evaluation sequences (Healthy reference vs Corrupted alternative)
    X_eval_ref = np.random.choice(4, size=(num_variant, seq_len)).astype(np.int8)
    X_eval_alt = X_eval_ref.copy()
    
    # At the mutation coordinate (512), substitute reference base with alternative base
    mutation_pos = seq_len // 2
    for i in range(num_variant):
        ref_base = X_eval_ref[i, mutation_pos]
        # Choose a different base for alternative mutation
        choices = [b for b in range(4) if b != ref_base]
        X_eval_alt[i, mutation_pos] = np.random.choice(choices)
        
    # Generate labels (1 = Pathogenic, 0 = Benign)
    Y_labels = np.random.choice([0, 1], size=(num_variant,)).astype(np.int8)
    
    # Save arrays
    np.save(os.path.join(dest_dir, "X_healthy.npy"), X_healthy)
    np.save(os.path.join(dest_dir, "X_corrupted.npy"), X_eval_alt)
    # We also save the original healthy reference windows for GVES calculation
    np.save(os.path.join(dest_dir, "X_healthy_ref.npy"), X_eval_ref)
    np.save(os.path.join(dest_dir, "Y_labels.npy"), Y_labels)

def preprocess_pipeline(config_path: str, dummy: bool = False):
    config = load_config(config_path)
    logger = setup_logger("ChromoDiff.Preprocess")
    
    data_dir = config.get("DATA_DIR", "data/processed")
    raw_dir = "data/raw"
    setup_dirs(data_dir, raw_dir)
    
    if dummy:
        logger.info("Generating synthetic dummy data for testing pipeline...")
        generate_dummy_data(data_dir, seed=config.get("SEED", 42))
        logger.info(f"Synthetic data saved successfully to {data_dir}!")
        return

    logger.info("Starting raw genomic data download and extraction...")
    
    # ClinVar download
    clinvar_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    clinvar_gz = os.path.join(raw_dir, "clinvar.vcf.gz")
    clinvar_vcf = os.path.join(raw_dir, "clinvar.vcf")
    
    try:
        download_file(clinvar_url, clinvar_gz, logger)
        extract_gzip(clinvar_gz, clinvar_vcf, logger)
    except Exception as e:
        logger.error(f"Failed to download/extract ClinVar: {e}")
        logger.info("Falling back to dummy mode.")
        generate_dummy_data(data_dir, seed=config.get("SEED", 42))
        return

    # hg38 download
    hg38_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    hg38_gz = os.path.join(raw_dir, "hg38.fa.gz")
    hg38_fa = os.path.join(raw_dir, "hg38.fa")
    
    try:
        download_file(hg38_url, hg38_gz, logger)
        extract_gzip(hg38_gz, hg38_fa, logger)
    except Exception as e:
        logger.error(f"Failed to download/extract hg38: {e}")
        logger.info("Falling back to dummy mode.")
        generate_dummy_data(data_dir, seed=config.get("SEED", 42))
        return

    # Importing Fasta here so it only triggers if pyfaidx is installed
    try:
        from pyfaidx import Fasta
    except ImportError:
        logger.error("pyfaidx library is missing. Install requirements.txt first.")
        return
        
    logger.info("Parsing files and generating token dataset...")
    # ClinVar parsing and reference extraction logic as in Section 3.2
    # We will write the full parsing logic
    try:
        genome = Fasta(hg38_fa, as_raw=True, sequence_always_upper=True)
    except Exception as e:
        logger.error(f"Error opening hg38.fa using pyfaidx: {e}")
        logger.info("Falling back to dummy data generation.")
        generate_dummy_data(data_dir, seed=config.get("SEED", 42))
        return

    # Parse ClinVar SNPs ONLY on testing chromosomes (chr21, chr22, chrX, chrY)
    logger.info("Parsing ClinVar SNPs on test chromosomes...")
    allowed_chroms = ["21", "22", "X", "Y"]
    rows = []
    
    with open(clinvar_vcf, "r") as f:
        for line in tqdm(f, desc="Parsing VCF"):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            info = parts[7]
            
            if chrom not in allowed_chroms:
                continue
            if len(ref) != 1:
                continue
            if "," in alt or len(alt) != 1:
                continue
            if ref not in NUC_TO_IDX or alt not in NUC_TO_IDX:
                continue

            mc_field = ""
            for field in info.split(";"):
                if field.startswith("MC="):
                    mc_field = field
                    break
            if any(consequence in mc_field for consequence in CODING_CONSEQUENCES):
                continue
                
            if "CLNSIG=Pathogenic" in info or "CLNSIG=Likely_pathogenic" in info:
                label = 1
            elif "CLNSIG=Benign" in info or "CLNSIG=Likely_benign" in info:
                label = 0
            else:
                continue
                
            rows.append((f"chr{chrom}", pos, ref, alt, label))
            
    df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "label"])
    logger.info(f"Found {len(df)} eligible non-coding SNPs.")
    
    # Extracted window processing
    WINDOW_SIZE = 1024
    MUT_POS = 512
    MAX_N_FRAC = 0.02
    
    ref_windows = []
    alt_windows = []
    labels = []
    
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting windows"):
        chrom = row.chrom
        if chrom not in genome.keys():
            continue
            
        pos1 = int(row.pos)
        start0 = pos1 - 1 - MUT_POS
        end0 = start0 + WINDOW_SIZE
        
        if start0 < 0:
            continue
            
        seq = genome[chrom][start0:end0]
        if len(seq) != WINDOW_SIZE:
            continue
            
        ref_tokens = seq_to_tokens(seq)
        r_idx = NUC_TO_IDX[row.ref]
        a_idx = NUC_TO_IDX[row.alt]
        
        if ref_tokens[MUT_POS] != r_idx:
            continue
            
        if (ref_tokens == 4).mean() > MAX_N_FRAC:
            continue
            
        alt_tokens = ref_tokens.copy()
        alt_tokens[MUT_POS] = a_idx
        
        ref_windows.append(ref_tokens)
        alt_windows.append(alt_tokens)
        labels.append(int(row.label))
        
    X_eval_ref = np.asarray(ref_windows, dtype=np.int8)
    X_eval_alt = np.asarray(alt_windows, dtype=np.int8)
    Y_labels = np.asarray(labels, dtype=np.int8)
    
    # Generate non-leaking pre-training data from hg38
    logger.info("Generating non-leaking pre-training data from hg38...")
    num_pretrain = int(config.get("NUM_PRETRAIN_WINDOWS", 200000))
    pretrain_seqs = []

    # Sample pre-training windows ONLY from training chromosomes (chr1-chr18) to prevent leakage.
    train_chroms = [f"chr{i}" for i in range(1, 19)]
    chroms = [k for k in genome.keys() if k in train_chroms]
    if len(chroms) == 0:
        chroms = list(genome.keys())

    np.random.seed(config.get("SEED", 42))

    pbar = tqdm(total=num_pretrain, desc="Sampling pre-training windows")
    attempts = 0
    max_attempts = num_pretrain * 10

    while len(pretrain_seqs) < num_pretrain and attempts < max_attempts:
        attempts += 1
        chrom = np.random.choice(chroms)
        chrom_len = len(genome[chrom])
        if chrom_len <= WINDOW_SIZE:
            continue

        start = np.random.randint(0, chrom_len - WINDOW_SIZE)
        end = start + WINDOW_SIZE

        seq = genome[chrom][start:end]
        if len(seq) != WINDOW_SIZE:
            continue

        ref_tokens = seq_to_tokens(seq)
        if (ref_tokens == 4).mean() > MAX_N_FRAC:
            continue

        pretrain_seqs.append(ref_tokens)
        pbar.update(1)

    pbar.close()

    X_healthy = np.asarray(pretrain_seqs, dtype=np.int8)
    
    # Save dataset arrays
    np.save(os.path.join(data_dir, "X_healthy.npy"), X_healthy)
    np.save(os.path.join(data_dir, "X_healthy_ref.npy"), X_eval_ref)
    np.save(os.path.join(data_dir, "X_corrupted.npy"), X_eval_alt)
    np.save(os.path.join(data_dir, "Y_labels.npy"), Y_labels)
    logger.info(f"Datasets generated successfully! Saved {len(X_healthy)} pre-training windows and {len(X_eval_ref)} ClinVar test windows to {data_dir}.")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/base_config.yaml"
    preprocess_pipeline(config_file, dummy=True)
