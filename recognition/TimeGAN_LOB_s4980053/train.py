# train.py - training and saving the baseline GAN and TimeGAN
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time

from dataset import make_loaders, TICKERS
from modules import BaselineGAN, TimeGAN, count_parameters
from utils import collect, invalid_rates, per_stock_report, windows_to_books
# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

BASELINE_EPOCHS = 50   # baseline GAN: settles by ~epoch 25
AE_EPOCHS = 100        # TimeGAN phase 1
SUP_EPOCHS = 500       # TimeGAN phase 2
JOINT_EPOCHS = 50      # TimeGAN phase 3
LR = 1e-3              # learning rate: how big each dial turn is (TimeGAN's paper uses 1e-3)
BATCH_SIZE = 128
SEQ_LEN = 500
HIDDEN_DIM = 24        # GRU memory size
NUM_LAYERS = 3
SEED = 42              # fixed randomness, so a run can be repeated exactly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    # GPU if there is one
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"   # default: smoke test. Real run: submit with SMOKE_TEST=0
MODEL = os.environ.get("MODEL", "timegan")              # "baseline" or "timegan", chosen when submitting

def bce(scores, real):
    """Binary cross-entropy against an all-'real' (1) or all-'fake' (0) target."""
    target = torch.ones_like(scores) if real else torch.zeros_like(scores)
    return F.binary_cross_entropy_with_logits(scores, target)


def moment_loss(fake, real):
    """How far the fake batch's average and spread are from the real batch's (per time step and feature)."""
    mean_gap = (fake.mean(dim=0) - real.mean(dim=0)).abs().mean()
    std_gap = (fake.std(dim=0) - real.std(dim=0)).abs().mean()
    return mean_gap + std_gap

# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train_baseline(model, loader, epochs, lr, device, max_batches=None):
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=lr)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        d_total, g_total, n = 0.0, 0.0, 0
        for real_window, stock in loader:
            real_window = real_window.to(device)
            stock = stock.to(device)
            noise = model.sample_noise(real_window.shape[0], real_window.shape[1], device)
            fake = model.make_fake(noise, stock)

            # discriminator catches fakes
            # generator dials → fake, so we fake.detach() → discriminator → fake_scores → d_loss
            real_scores = model.judge(real_window, stock)
            fake_scores = model.judge(fake.detach(), stock)
            d_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores)) \
                   + F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
            d_opt.zero_grad() 
            d_loss.backward() 
            d_opt.step() 

            # generator's turn: fool the discriminator
            # generator dials → fake → discriminator → g_scores → g_loss -> loop backward() to generator dials
            g_scores = model.judge(fake, stock)  # NOT detached: the trace must reach the generator
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
        for real, stock in loader:
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

def train_supervisor(model, loader, epochs, lr, device, max_batches=None):
    """TimeGAN phase 2: the supervisor learns to predict the next latent code. Embedder frozen."""
    opt = torch.optim.Adam(model.supervisor.parameters(), lr=lr)              # ONLY the supervisor's dials
    history = []

    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for real, stock in loader:
            real = real.to(device)
            stock = stock.to(device)
            with torch.no_grad():                      # frozen: no trace into the embedder
                codes = model.embedder(real)                            # zip the real window
            guesses = model.supervise(codes, stock)                              # supervisor on the codes
            loss = F.mse_loss(guesses[:, :-1], codes[:, 1:])   # guess for t+1 vs the real code at t+1
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break

        history.append(total / n)
        print(f"[supervisor] epoch {epoch}/{epochs}  next-step mse {total / n:.4f}")

    return history

def train_joint(model, loader, epochs, lr, device, max_batches=None, gamma=1.0):
    """TimeGAN phase 3: generator, supervisor, zip tool and discriminator train together."""
    g_opt = torch.optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=lr)
    e_opt = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    history = {"g_loss": [], "e_loss": [], "d_loss": []}

    for epoch in range(1, epochs + 1):
        g_total, e_total, d_total, n = 0.0, 0.0, 0.0, 0
        for real, stock in loader:
            real = real.to(device)
            stock = stock.to(device)

            for _ in range(2):                                     # turns 1 and 2 happen twice
                # --- turn 1: generator + supervisor ---
                noise = model.sample_noise(real.shape[0], real.shape[1], device)
                fake_raw = model.make_codes(noise, stock)           # generator's fake codes
                fake = model.supervise(fake_raw, stock)             # after the supervisor
                fake_windows = model.recovery(fake)                 # unzipped into features
                real_codes = model.embedder(real)
                adv = bce(model.judge(fake, stock), True) + gamma * bce(model.judge(fake_raw, stock), True)
                sup = F.mse_loss(model.supervise(real_codes, stock)[:, :-1], real_codes[:, 1:])
                g_loss = adv + 100 * torch.sqrt(sup) + 100 * moment_loss(fake_windows, real)    # moment loss between fake_windows and real
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()                                                 # g_opt: step

                # --- turn 2: "zip tool" ---
                real_codes = model.embedder(real)
                recon = F.mse_loss(model.recovery(real_codes), real)                                         # mse between recovery(real_codes) and real
                sup = F.mse_loss(model.supervise(real_codes, stock)[:, :-1], real_codes[:, 1:])
                e_loss = 10 * torch.sqrt(recon) + 0.1 * sup
                e_opt.zero_grad()
                e_loss.backward()
                e_opt.step()

            # --- turn 3: discriminator (once per batch) ---
            with torch.no_grad():                                   # the judge's turn: nothing else learns
                real_codes = model.embedder(real)
                fake_raw = model.make_codes(noise, stock)
                fake = model.supervise(fake_raw, stock)
            d_loss = bce(model.judge(real_codes, stock), True) + bce(model.judge(fake, stock), False) \
                   + gamma * bce(model.judge(fake_raw, stock), False)
            if d_loss.item() > 0.15:                                # skip if already winning
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

            g_total += g_loss.item()
            e_total += e_loss.item()
            d_total += d_loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break

        for key, total in (("g_loss", g_total), ("e_loss", e_total), ("d_loss", d_total)):
            history[key].append(total / n)
        print(f"[joint] epoch {epoch}/{epochs}  g {g_total / n:.4f}  e {e_total / n:.4f}  d {d_total / n:.4f}")

    return history

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)

    # data (shared by both models)
    train_loader, val_loader, test_loader, mean, std = make_loaders(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)    
    feature_dim = mean.shape[0]                                # 41: one mean per feature
    max_batches = 5 if SMOKE_TEST else None

    if MODEL == "baseline":
        model = BaselineGAN(feature_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
        print(f"device: {DEVICE} | parameters: {count_parameters(model)}")

        start = time.time()
        history = train_baseline(model, train_loader, 1 if SMOKE_TEST else BASELINE_EPOCHS, LR, DEVICE, max_batches)
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

        real_w, real_l = collect(test_loader)                           # all 315 test windows
        fake_l = torch.arange(1000) % 5                                 # 200 fake windows per stock
        fake_w = model.generate(fake_l.to(DEVICE), SEQ_LEN, DEVICE).cpu().numpy()
        per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)

    else:
        model = TimeGAN(feature_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
        print(f"device: {DEVICE} | parameters: {count_parameters(model)}")

        start = time.time()
        ae_history = train_autoencoder(model, train_loader, 1 if SMOKE_TEST else AE_EPOCHS, LR, DEVICE, max_batches)
        supervisor_history = train_supervisor(model, train_loader, 1 if SMOKE_TEST else SUP_EPOCHS, LR, DEVICE, max_batches)
        joint_history = train_joint(model, train_loader, 1 if SMOKE_TEST else JOINT_EPOCHS, LR, DEVICE, max_batches)   # NEW: phase 3
        print(f"training time: {(time.time() - start) / 60:.1f} min")
        if DEVICE == "cuda":                                                                                       # NEW: criterion 5
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

        # NEW: save the model, plus the settings predict.py will need to rebuild it
        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "settings": {"hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN}},
                "timegan.pt")

        # NEW: phase 3 loss curves
        plt.figure()
        for key in ("g_loss", "e_loss", "d_loss"):
            plt.plot(joint_history[key], label=key)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("timegan_losses.png", dpi=150)

        # check phase 1 on TEST windows: hours the autoencoder never trained on
        real, stock = next(iter(test_loader))
        real = real.to(DEVICE)
        stock = stock.to(DEVICE)
        with torch.no_grad():
            rebuilt = model.reconstruct(real)
        print(f"test reconstruction mse: {F.mse_loss(rebuilt, real).item():.4f}")
        books = windows_to_books(rebuilt.cpu().numpy(), mean, std)
        spread, ladder, volume = invalid_rates(books)
        print(f"rebuilt books broken - spread: {spread:.1%}  ladder: {ladder:.1%}  volume: {volume:.1%}")

        # check phase 2 on TEST windows
        with torch.no_grad():
            codes = model.embedder(real)
            guesses = model.supervise(codes, stock)
        print(f"test next-step mse: {F.mse_loss(guesses[:, :-1], codes[:, 1:]).item():.2e}")
        print(f"'no change' guess mse: {F.mse_loss(codes[:, :-1], codes[:, 1:]).item():.2e}")

        # NEW: the real test: generated books, per stock (same report as the baseline)
        real_w, real_l = collect(test_loader)
        fake_l = torch.arange(1000) % 5
        fake_w = model.generate(fake_l.to(DEVICE), SEQ_LEN, DEVICE).cpu().numpy()
        per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)