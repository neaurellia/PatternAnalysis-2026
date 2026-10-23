# modules.py - the networks

import torch
import torch.nn as nn
import torch.nn.functional as F

def with_label(x, labels, num_stocks=5):
    """Attach each window's one-hot stock label to every time step: (B, T, d) -> (B, T, d + num_stocks)."""
    onehot = F.one_hot(labels, num_stocks).float()                                              # (B,) labels -> (B, num_stocks) switches
    onehot = onehot[:, None, :].expand(-1, x.shape[1], -1)    # copy them to every time step -> (B, T, num_stocks)
    return torch.cat([x, onehot], dim=-1)                                                # glue onto x along the LAST axis

# ---------------------------------------------------------------------------
# RecurrentBlock - the shared building block
# shape: (batch, 60, in_dim) -> (batch, 60, out_dim)
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, sigmoid=False):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.sigmoid = sigmoid

    def forward(self, x):
        output, _ = self.gru(x)
        out = self.linear(output)
        if self.sigmoid:
            out = torch.sigmoid(out)                      # squash into 0-1 
        return out

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

class TimeGAN(nn.Module):
    """TimeGAN (Yoon et al., 2019). Part 1: the autoencoder (embedder + recovery) and supervisor for now."""

    def __init__(self, feature_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedder = RecurrentBlock(feature_dim, hidden_dim, hidden_dim, num_layers, sigmoid=True)     # zip:   feature_dim in -> hidden_dim out, WITH sigmoid
        self.recovery = RecurrentBlock(hidden_dim, hidden_dim, feature_dim, num_layers)     # unzip: hidden_dim in -> feature_dim out, no sigmoid
        self.supervisor = RecurrentBlock(hidden_dim, hidden_dim, hidden_dim, num_layers, sigmoid=True)

    def reconstruct(self, x):
        """Zip then unzip a real window. Used to train phase 1."""
        code = self.embedder(x)
        return self.recovery(code)              # x -> embedder -> recovery

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
    labels = torch.tensor([0, 2, 4])
    tagged = with_label(torch.randn(3, 500, 41), labels)
    print("tagged:", tagged.shape)              # expect (3, 500, 46)
    print("window 1, step 0, switches:", tagged[1, 0, 41:])   # expect GOOG: [0, 0, 1, 0, 0]

    block = RecurrentBlock(in_dim=41, hidden_dim=24, out_dim=41, num_layers=3)
    x = torch.randn(128, 60, 41)
    print(block(x).shape)

    gan = BaselineGAN(feature_dim=41, hidden_dim=24, num_layers=3)
    fake = gan.generate(batch_size=8, seq_len=60, device="cpu")
    print("fake windows:", fake.shape)                  # expect (8, 60, 41)
    print("scores:", gan.discriminator(fake).shape)     # expect (8, 60, 1)

    print("parameters: ", count_parameters(gan))

    squashed = RecurrentBlock(41, 24, 24, 3, sigmoid=True)(x)
    print("sigmoid block:", squashed.shape, "min", squashed.min().item(), "max", squashed.max().item())

    tg = TimeGAN(feature_dim=41, hidden_dim=24, num_layers=3)
    print("reconstruction:", tg.reconstruct(x).shape)                   # expect (128, 60, 41)
    print("autoencoder parameters:", count_parameters(tg))

    codes = tg.embedder(x)                # real-ish window -> zip files
    print("codes:", codes.shape)              
    print("supervisor:", tg.supervisor(codes).shape)   # expect (128, 60, 24): one guess per step