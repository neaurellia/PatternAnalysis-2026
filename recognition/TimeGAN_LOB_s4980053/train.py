# train.py - training and saving the baseline GAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time

from dataset import make_loaders
from modules import BaselineGAN, TimeGAN, count_parameters
from utils import invalid_rates, windows_to_books, vwap_stats, volatility_clustering

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

EPOCHS = 50            # passes over the whole training set
LR = 1e-3              # learning rate: how big each dial turn is (TimeGAN's paper uses 1e-3)
BATCH_SIZE = 128
SEQ_LEN = 60
HIDDEN_DIM = 24        # GRU memory size
NUM_LAYERS = 3
SEED = 42              # fixed randomness, so a run can be repeated exactly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    # GPU if there is one
SMOKE_TEST = True      # True: 1 epoch, 5 batches, just to check it runs. False: real training.
MODEL = "timegan"      # "baseline" or "timegan"

# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train_baseline(model, loader, epochs, lr, device, max_batches=None):
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=lr)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        d_total, g_total, n = 0.0, 0.0, 0
        for real_window in loader:
            real_window = real_window.to(device)
            noise = model.sample_noise(real_window.shape[0], real_window.shape[1], device)
            fake = model.generator(noise)

            # discriminator catches fakes
            # generator dials → fake, so we fake.detach() → discriminator → fake_scores → d_loss
            real_scores = model.discriminator(real_window)
            fake_scores = model.discriminator(fake.detach())
            d_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores)) \
                   + F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
            d_opt.zero_grad() 
            d_loss.backward() 
            d_opt.step() 

            # generator's turn: fool the discriminator
            # generator dials → fake → discriminator → g_scores → g_loss -> loop backward() to generator dials
            g_scores = model.discriminator(fake)  # NOT detached: the trace must reach the generator
            g_loss = F.binary_cross_entropy_with_logits(g_scores, torch.ones_like(g_scores))                          # g_scores vs target 1
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            d_total += d_loss.item()
            g_total += g_loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break                             # smoke test: stop early

        history["d_loss"].append(d_total / n)
        history["g_loss"].append(g_total / n)
        print(f"epoch {epoch}/{epochs}  d_loss {d_total / n:.4f}  g_loss {g_total / n:.4f}")

    return history

def train_autoencoder(model, loader, epochs, lr, device, max_batches=None):
    """TimeGAN phase 1: embedder + recovery learn to zip and unzip real windows."""
    '''for each batch of real windows:
        rebuilt = reconstruct(real)
        loss = how different rebuilt is from real   (mean squared error)
        zero_grad → backward → step'''

    params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for real in loader:
            real = real.to(device)
            rebuilt = model.reconstruct(real)                           # zip + unzip (your new method)
            loss = F.mse_loss(rebuilt, real)        # mean squared error
            opt.zero_grad()
            loss.backward()
            opt.step()                                     # the three optimiser lines

            total += loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break

        history.append(total / n)
        print(f"[autoencoder] epoch {epoch}/{epochs}  reconstruction mse {total / n:.4f}")

    return history

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)
    book_file = "data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    msg_file = "data/AMZN_2012-06-21_34200000_57600000_message_10.csv"

    # data (shared by both models)
    train_loader, val_loader, test_loader, mean, std = make_loaders(book_file, msg_file, SEQ_LEN, BATCH_SIZE)
    feature_dim = mean.shape[0]                                # 41: one mean per feature
    epochs, max_batches = (1, 5) if SMOKE_TEST else (EPOCHS, None)

    if MODEL == "baseline":
        model = BaselineGAN(feature_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
        print(f"device: {DEVICE} | parameters: {count_parameters(model)}")

        start = time.time()
        history = train_baseline(model, train_loader, epochs, LR, DEVICE, max_batches)
        minutes = (time.time() - start) / 60

        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "settings": {"hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN}},
                   "baseline.pt")

        plt.plot(history["d_loss"], label="discriminator")
        plt.plot(history["g_loss"], label="generator")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("baseline_losses.png", dpi=150)

        print(f"training time: {minutes:.1f} min")
        if DEVICE == "cuda":
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

        fake = model.generate(1000, SEQ_LEN, DEVICE).cpu().numpy()
        books = windows_to_books(fake, mean, std)
        spread, ladder, volume = invalid_rates(books)
        print(f"broken books - spread: {spread:.1%}  ladder: {ladder:.1%}  volume: {volume:.1%}")

        real = next(iter(test_loader)).numpy()
        real_books = windows_to_books(real, mean, std)
        print("VWAP offset (mean, std) - real:", vwap_stats(real_books), " fake:", vwap_stats(books))
        print(f"volatility clustering - real: {volatility_clustering(real_books):.3f}  fake: {volatility_clustering(books):.3f}")

    else:
        model = TimeGAN(feature_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)                                        # a TimeGAN with feature_dim, HIDDEN_DIM, NUM_LAYERS, on DEVICE
        print(f"device: {DEVICE} | parameters: {count_parameters(model)}")

        start = time.time()
        ae_history = train_autoencoder(model, train_loader, epochs, LR, DEVICE, max_batches)                                   # call train_autoencoder (same arguments as train_baseline)
        print(f"training time: {(time.time() - start) / 60:.1f} min")

        # check phase 1 on TEST windows: hours the autoencoder never trained on
        real = next(iter(test_loader)).to(DEVICE)
        with torch.no_grad():
            rebuilt = model.reconstruct(real)
        print(f"test reconstruction mse: {F.mse_loss(rebuilt, real).item():.4f}")

        books = windows_to_books(rebuilt.cpu().numpy(), mean, std)
        spread, ladder, volume = invalid_rates(books)
        print(f"rebuilt books broken - spread: {spread:.1%}  ladder: {ladder:.1%}  volume: {volume:.1%}")

        torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "timegan.pt")