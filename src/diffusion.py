import torch

class AbsorbingStateScheduler:
    """
    Scheduler for absorbing-state discrete genomic diffusion (D3PM-Absorb).
    Defines beta/alpha cumulative product schedules and injects noise by replacing tokens with [MASK] (5).
    """
    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, min_corruption_rate: float = 0.15):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.min_corruption_rate = min_corruption_rate

        # Initialize linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Importance-weighted timestep sampling.
        Uses a squared distribution to bias toward low/mid corruption levels
        where the model can actually learn sequence context from surrounding bases.
        High t (>800) masks >90% of tokens, leaving no context to learn from.
        """
        u = torch.rand(batch_size, device=device)
        t = (u ** 2 * self.num_steps).long().clamp(0, self.num_steps - 1)
        return t

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample noisy sequence x_t from clean x_0.
        
        Each base either:
          - survives unchanged   with prob ᾱ_t  (but never more than 1 - min_corruption_rate)
          - is replaced by [MASK] (5) with prob max(1 - ᾱ_t, min_corruption_rate)
        """
        B, L = x_start.shape
        device = x_start.device
        
        if self.alphas_cumprod.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        # Get alphas_cumprod for the given step
        a_bar = self.alphas_cumprod[t].unsqueeze(1) # [B, 1]

        # Clamp survival probability so corruption rate (masking) >= min_corruption_rate (e.g. 15%)
        a_bar_floored = torch.clamp(a_bar, max=1.0 - self.min_corruption_rate)

        rand_probs = torch.rand((B, L), device=device)
        mutate_mask = rand_probs > a_bar_floored # [B, L] Bool

        # Replace selected bases with [MASK] (5)
        x_noisy = torch.where(mutate_mask, torch.tensor(5, device=device, dtype=torch.long), x_start)

        return x_noisy, mutate_mask
