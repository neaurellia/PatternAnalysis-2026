# modules.py - the networks

# TODO: imports (torch, torch.nn)


# ---------------------------------------------------------------------------
# RecurrentBlock - the shared building block
# shape: (batch, 60, in_dim) -> (batch, 60, out_dim)
# ---------------------------------------------------------------------------
# TODO: class RecurrentBlock(nn.Module)
# TODO:   __init__(in_dim, hidden_dim, out_dim, num_layers)
# TODO:     a GRU: in_dim -> hidden_dim, num_layers deep, batches first
# TODO:     a Linear layer: hidden_dim -> out_dim
# TODO:   forward(x)
# TODO:     run the GRU, keep the output at EVERY step (not just the last)
# TODO:     run the Linear layer on that, return it


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
# TODO: if __name__ == "__main__":
# TODO:   RecurrentBlock on random (128, 60, 41) -> expect (128, 60, 41)
# TODO:   BaselineGAN(41, ...): generate 8 windows -> expect (8, 60, 41)
# TODO:   discriminator on those windows -> expect (8, 60, 1)
# TODO:   print the parameter count


# LATER (after the baseline trains): TimeGAN - autoencoder, supervisor, generator/discriminator
