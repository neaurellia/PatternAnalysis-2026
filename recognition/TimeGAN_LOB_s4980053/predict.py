# predict.py - example usage of a trained model: load it, generate order books, score them, plot them

import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import TICKERS, make_loaders
from modules import BaselineGAN, TimeGAN
from utils import collect, per_stock_report, vwap_offset, windows_to_books


def load_model(kind, path, feature_dim=41):
    """Rebuild a saved model. kind is "baseline" or "timegan". Returns the model, mean, std, settings."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    s = ckpt["settings"]
    model_class = BaselineGAN if kind == "baseline" else TimeGAN
    model = model_class(feature_dim, s["hidden_dim"], s["num_layers"])
    model.load_state_dict(ckpt["model"])                     # pour the saved dials into it
    model.eval()
    return model, ckpt["mean"], ckpt["std"], s

def plot_vwap_over_time(real_w, real_l, fake_w, fake_l, mean, std, path, n=3):
    """One panel per stock: VWAP offset over time for the first n real and first n fake windows."""
    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(10, 12), sharex=True)
    for stock_id, (ticker, ax) in enumerate(zip(TICKERS, axes)):
        real_books = windows_to_books(real_w[real_l == stock_id][:n], mean, std)
        fake_books = windows_to_books(fake_w[fake_l == stock_id][:n], mean, std)                                       # same, for this stock's first n fake windows
        for i in range(n):
            ax.plot(vwap_offset(real_books[i]), color="tab:blue", alpha=0.7, label="real" if i == 0 else None)
            ax.plot(vwap_offset(fake_books[i]), color="tab:orange", alpha=0.7, label="fake" if i == 0 else None)
        ax.set_title(ticker)
        ax.set_ylabel("VWAP - mid (ticks)")
    axes[-1].set_xlabel("time step (seconds)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_vwap_per_stock(real_w, real_l, fake_w, fake_l, mean, std, path):
    """Mean +/- std of the VWAP offset per stock, real next to fake."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for stock_id in range(len(TICKERS)):
        for windows, labels, shift, color, name in ((real_w, real_l, -0.15, "tab:blue", "real"),
                                                    (fake_w, fake_l, 0.15, "tab:orange", "fake")):
            v = vwap_offset(windows_to_books(windows[labels == stock_id], mean, std))
            ax.errorbar(stock_id + shift, v.mean(), yerr=v.std(), fmt="o", color=color, capsize=4,
                        label=name if stock_id == 0 else None)
    ax.set_xticks(range(len(TICKERS)), TICKERS)
    ax.set_ylabel("VWAP - mid (ticks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]          # e.g. python predict.py baseline baseline_trained.pt
    model, mean, std, settings = load_model(kind, path)

    _, _, test_loader, _, _ = make_loaders(seq_len=settings["seq_len"])     # only the real test windows are needed
    real_w, real_l = collect(test_loader)

    fake_l = torch.arange(1000) % 5
    fake_w = model.generate(fake_l, settings["seq_len"], "cpu").numpy()
    per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)

    plot_vwap_over_time(real_w, real_l, fake_w, fake_l.numpy(), mean, std, f"{kind}_vwap_over_time.png")
    plot_vwap_per_stock(real_w, real_l, fake_w, fake_l.numpy(), mean, std, f"{kind}_vwap_per_stock.png")