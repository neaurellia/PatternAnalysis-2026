# predict.py - example usage of a trained model: load it, generate order books, score them, plot them

import sys

import numpy as np
import torch

from dataset import TICKERS, make_loaders
from modules import BaselineGAN, TimeGAN
from utils import collect, per_stock_report


def load_model(kind, path, feature_dim=41):
    """Rebuild a saved model. kind is "baseline" or "timegan". Returns the model, mean, std, settings."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    s = ckpt["settings"]
    model_class = BaselineGAN if kind == "baseline" else TimeGAN
    model = model_class(feature_dim, s["hidden_dim"], s["num_layers"])
    model.load_state_dict(ckpt["model"])                     # pour the saved dials into it
    model.eval()
    return model, ckpt["mean"], ckpt["std"], s


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]          # e.g. python predict.py baseline baseline_trained.pt
    model, mean, std, settings = load_model(kind, path)

    _, _, test_loader, _, _ = make_loaders(seq_len=settings["seq_len"])     # only the real test windows are needed
    real_w, real_l = collect(test_loader)

    fake_l = torch.arange(1000) % 5
    fake_w = model.generate(fake_l, settings["seq_len"], "cpu").numpy()
    per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)