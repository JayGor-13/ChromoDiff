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

---

## Quick Start

### Train
```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluate (AUROC / AUPRC)
```bash
python scripts/evaluate.py \
    --config  configs/default.yaml \
    --weights outputs/checkpoints/default/best.pth \
    --data    data/X_corrupted.npy \
    --labels  data/Y_labels.npy
```

### Score a single variant (Python API)
```python
import torch
from chromodiff.model.genodiff import GenoDiff1D
from chromodiff.diffusion.schedule import make_beta_schedule
from chromodiff.scoring.gves import calculate_gves

device = torch.device("cuda")
model  = GenoDiff1D().to(device)
model.load_state_dict(torch.load("outputs/checkpoints/default/best.pth"))

_, _, alphas_cumprod = make_beta_schedule(1000, 1e-4, 0.02, device)

# Your 1024-bp genomic window as a Long tensor
seq = torch.tensor([your_sequence], dtype=torch.long).unsqueeze(0)

score = calculate_gves(model, seq, alphas_cumprod,
                       mutation_pos=512, ref_base="A", alt_base="T")
print(f"GVES: {score:.4f}")  # positive = pathogenic
```

---

## Project Structure

```
ChromoDiff/
├── configs/           # All hyperparameters (YAML) — one file per experiment
├── chromodiff/        # Installable Python package
│   ├── data/          # Dataset & DataLoader
│   ├── diffusion/     # Beta schedule + q_sample
│   ├── model/         # Embeddings, ResBlocks, GenoDiff1D
│   ├── scoring/       # GVES, sequence_anomaly_energy, score_dataset
│   └── interpretability/  # Input gradient saliency maps
├── scripts/           # CLI entry points (train, evaluate, ablations)
├── notebooks/         # Exploratory Jupyter notebooks
└── tests/             # Pytest unit tests
```

---

## Architecture

| Component | Detail |
|-----------|--------|
| Backbone | 1D Dilated Residual CNN |
| Hidden dim | 256 |
| Blocks | 8 × DilatedResidualBlock |
| Dilations | 1, 2, 4, 8, 16, 32, 64, 128 |
| Receptive field | ~1017 bp (covers full window) |
| Parameters | ~4.8M |
| Normalisation | GroupNorm (16 groups) |
| Time conditioning | Sinusoidal embeddings → 4x MLP |

---

## Running Tests

```bash
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
