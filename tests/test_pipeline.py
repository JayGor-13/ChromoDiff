import pytest
import torch
import numpy as np

from src.models.embedding import SinusoidalPositionEmbeddings
from src.models.unet import DilatedResidualBlock, GenoDiff1D
from src.diffusion import AbsorbingStateScheduler
from src.dataset import GenomicDataset, reverse_complement_tokens, reverse_complement_tensor

def test_sinusoidal_embeddings():
    dim = 256
    embedder = SinusoidalPositionEmbeddings(dim)
    t = torch.tensor([0, 10, 100, 999], dtype=torch.long)
    emb = embedder(t)
    
    assert emb.shape == (4, dim)
    assert not torch.isnan(emb).any()
    assert emb.dtype == torch.float32

def test_dilated_block():
    hidden_dim = 256
    dilation = 4
    block = DilatedResidualBlock(hidden_dim, dilation)
    
    x = torch.randn(4, hidden_dim, 1024)
    t_emb = torch.randn(4, hidden_dim)
    
    out = block(x, t_emb)
    
    assert out.shape == (4, hidden_dim, 1024)
    assert not torch.isnan(out).any()

def test_genodiff_model():
    vocab_size = 6
    hidden_dim = 256
    model = GenoDiff1D(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
    x = torch.randint(0, vocab_size, (2, 1024), dtype=torch.long)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    
    logits = model(x, t)
    
    assert logits.shape == (2, vocab_size - 1, 1024)
    assert not torch.isnan(logits).any()

def test_diffusion_scheduler():
    num_steps = 1000
    scheduler = AbsorbingStateScheduler(
        num_steps=num_steps,
        beta_start=1e-4,
        beta_end=0.02,
        min_corruption_rate=0.15
    )
    
    # Test alphas_cumprod shape
    assert scheduler.alphas_cumprod.shape == (num_steps,)
    
    # Test q_sample shape and bounds
    x_start = torch.randint(0, 5, (16, 1024), dtype=torch.long)
    t_ones = torch.ones(16, dtype=torch.long)
    x_noisy, mutate_mask = scheduler.q_sample(x_start, t_ones)
    
    assert x_noisy.shape == (16, 1024)
    assert mutate_mask.shape == (16, 1024)
    assert x_noisy.dtype == torch.long
    
    # Check that mask values are bool
    assert mutate_mask.dtype == torch.bool
    
    # Check that mutated positions contain the MASK token (5)
    masked_positions = x_noisy[mutate_mask]
    if len(masked_positions) > 0:
        assert torch.all(masked_positions == 5)
        
    # Check that unmutated positions equal the start positions
    assert torch.all(x_noisy[~mutate_mask] == x_start[~mutate_mask])
    
    # Test minimum corruption rate floor
    t_zeros = torch.zeros(32, dtype=torch.long)
    x_start_32 = torch.randint(0, 5, (32, 1024), dtype=torch.long)
    _, mutate_mask_0 = scheduler.q_sample(x_start_32, t_zeros)
    pct_corrupted = mutate_mask_0.float().mean().item()
    
    # Assert corruption rate is close to or greater than the 15% floor (using a small margin for randomness)
    assert pct_corrupted >= 0.12 # stochastic check

def test_genomic_dataset():
    data = torch.randint(0, 5, (10, 1024))
    dataset = GenomicDataset(data)
    
    assert len(dataset) == 10
    assert dataset[0].shape == (1024,)
    assert dataset[0].dtype == torch.long

def test_reverse_complement():
    # Numpy array RC
    seq_np = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int8) # A, C, G, T, N, [MASK]
    rc_np = reverse_complement_tokens(seq_np)
    # Expected: reversed order and complemented mapping
    # Original order reversed: 5, 4, 3, 2, 1, 0
    # Mapping: 5->5, 4->4, 3->0, 2->1, 1->2, 0->3
    # Expected: 5, 4, 0, 1, 2, 3
    expected_np = np.array([[5, 4, 0, 1, 2, 3]], dtype=np.int8)
    assert np.array_equal(rc_np, expected_np)
    
    # PyTorch Tensor RC
    seq_tensor = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    rc_tensor = reverse_complement_tensor(seq_tensor)
    expected_tensor = torch.tensor([[5, 4, 0, 1, 2, 3]], dtype=torch.long)
    assert torch.equal(rc_tensor, expected_tensor)
