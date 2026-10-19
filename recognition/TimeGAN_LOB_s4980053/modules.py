# modules.py - the networks

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RecurrentBlock - the shared building block
# shape: (batch, 60, in_dim) -> (batch, 60, out_dim)
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.linear(output)

# ---------------------------------------------------------------------------
# BaselineGAN - a plain GAN for sequences
# ---------------------------------------------------------------------------

class BaselineGAN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super().__init__()
        self.noise_dim = feature_dim
        self.generator = RecurrentBlock(self.noise_dim, hidden_dim, feature_dim, num_layers)
        self.discriminator = RecurrentBlock(feature_dim, hidden_dim, 1, num_layers)

    def sample_noise(self, batch_size, seq_len, device):
        return torch.randn(batch_size, seq_len, self.noise_dim, device=device)

    @torch.no_grad()
    def generate(self, batch_size, seq_len, device):
        noise = self.sample_noise(batch_size, seq_len, device)
        return self.generator(noise)

def count_parameters(model):
    """Number of trainable weights in a model."""
    total = 0
    for p in model.parameters():    # p = one block of weights at a time
        total += p.numel()          # add how many numbers are in it
    return total

# ---------------------------------------------------------------------------
# quick self-test: python modules.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    block = RecurrentBlock(in_dim=41, hidden_dim=24, out_dim=41, num_layers=3)
    x = torch.randn(128, 60, 41)
    print(block(x).shape)

    gan = BaselineGAN(feature_dim=41, hidden_dim=24, num_layers=3)
    fake = gan.generate(batch_size=8, seq_len=60, device="cpu")
    print("fake windows:", fake.shape)                  # expect (8, 60, 41)
    print("scores:", gan.discriminator(fake).shape)     # expect (8, 60, 1)

    print("parameters: ", count_parameters(gan))


# LATER (after the baseline trains): TimeGAN - autoencoder, supervisor, generator/discriminator
