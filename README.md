# ChromoDiff 🧬

**Generative Zero-Shot Pathogenicity Prediction via Discrete Genomic Diffusion**

ChromoDiff is an unsupervised deep learning model that identifies cancer-driving (pathogenic) DNA mutations in the **non-coding genome** — the 98% of the genome where protein-folding models like AlphaMissense fail.

Instead of training a supervised classifier on biased cancer databases, ChromoDiff learns the natural biological manifold of **healthy human DNA (hg38)**. At inference time, mutations are treated as out-of-distribution anomalies: sequences that the model finds hard to reconstruct receive high anomaly energy scores.

---

## How It Works

```
Healthy DNA (hg38) ──► Forward Diffusion (add noise) ──► Denoiser ──► Learns genomic grammar
                                                                              │
                                                                              ▼
Cancer Variant ──────────────────────────────────────────────► High NLL (anomaly) = PATHOGENIC
Benign Variant ──────────────────────────────────────────────► Low NLL  (normal)  = BENIGN
```

**GVES Score** = `NLL(alt-base sequence) − NLL(ref-base sequence)`
- Positive → model finds alt-base anomalous → pathogenic signal
- Near zero → benign / ambiguous

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/chromodiff.git
cd chromodiff
pip install -e .
```

## Quick Start

### 1. Run Preprocessing
To extract sequences from the human genome (`hg38.fa`) and ClinVar mutations (`clinvar.vcf`):
```bash
python run_pipeline.py --mode preprocess
```
*(For testing, pass `--dummy` to instantly generate high-quality synthetic data).*

### 2. Train Model
```bash
python run_pipeline.py --mode train
```

### 3. Evaluate Zero-Shot GVES Pathogenicity (AUROC/AUPRC)
```bash
python run_pipeline.py --mode evaluate --weights outputs/checkpoints/genodiff_best.pth
```

### 4. Score a Single Variant (Python API)
```python
import torch
from src.models.unet import GenoDiff1D
from src.evaluate import calculate_gves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenoDiff1D(vocab_size=6, hidden_dim=256).to(device)
model.load_state_dict(torch.load("outputs/checkpoints/genodiff_best.pth", map_location=device))

# Sequence token array [1, 1024] (A=0, C=1, G=2, T=3, N=4, [MASK]=5)
seq = torch.randint(0, 5, (1, 1024), dtype=torch.long).to(device)

score = calculate_gves(model, seq, ref_base="A", alt_base="T", mutation_pos=512)
print(f"GVES: {score[0]:.4f}") # Positive = pathogenic, Near-zero = benign
```

---

## Project Structure

```
ChromoDiff/
├── configs/                  # Experiment configuration files
│   └── base_config.yaml      # Hyperparameters (learning rate, hidden_dim, batch_size)
│
├── src/                      # Main source code directory
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset and RC Augmentation
│   ├── diffusion.py          # AbsorbingStateScheduler (Noise logic)
│   ├── preprocess.py         # ClinVar & Fasta data preprocessor
│   ├── train.py              # Denoising diffusion training loop
│   ├── evaluate.py           # Zero-shot GVES scoring and metrics evaluation
│   ├── utils.py              # Logging, seeds, directory builders, and config helpers
│   └── models/               # Model architecture modules
│       ├── __init__.py
│       ├── embedding.py      # Sinusoidal time embedding module
│       └── unet.py           # Dilated blocks and GenoDiff1D model
│
├── tests/                    # Pytest unit tests
│   └── test_pipeline.py      # Pytest covering all network components and scheduler
│
├── chromodiff_kaggle.ipynb   # Self-contained Kaggle-ready notebook (writes code + trains on P100 GPU)
├── run_pipeline.py           # Single main CLI entry point
└── requirements.txt          # Python package requirements
```

---

## Running Tests

To run the automated validation test suite:
```bash
$env:PYTHONPATH="."
pytest tests/ -v
```

---

## Citation

```bibtex
@misc{chromodiff2026,
  title  = {ChromoDiff: Generative Zero-Shot Pathogenicity Prediction via Discrete Genomic Diffusion},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/chromodiff}
}
```
