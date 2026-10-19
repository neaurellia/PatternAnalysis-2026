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
# TODO: class BaselineGAN(nn.Module)
# TODO:   __init__(feature_dim, hidden_dim, num_layers)
# TODO:     remember the noise size (same as feature_dim keeps it simple)
# TODO:     generator: a RecurrentBlock, noise -> features
# TODO:     discriminator: a RecurrentBlock, features -> 1 score per time step
# TODO:   sample_noise(batch_size, seq_len, device)
# TODO:     random numbers shaped (batch_size, seq_len, noise size), on the right device
# TODO:   generate(batch_size, seq_len, device)
# TODO:     noise -> generator -> fake windows
# TODO:     no gradients needed here (it's only used after training) - how do you turn them off?


# TODO: count_parameters(model)
# TODO:   add up the number of trainable weights


# ---------------------------------------------------------------------------
# quick self-test: python modules.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    block = RecurrentBlock(in_dim=41, hidden_dim=24, out_dim=41, num_layers=3)
    x = torch.randn(128, 60, 41)
    print(block(x).shape)


# LATER (after the baseline trains): TimeGAN - autoencoder, supervisor, generator/discriminator
