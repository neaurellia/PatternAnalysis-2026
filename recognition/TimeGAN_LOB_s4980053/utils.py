# utils.py - scoring generated order books against real ones
# (all of these work on DECODED books, not on normalised features)

import numpy as np
import torch

from dataset import check_rules, decode

def windows_to_books(windows, mean, std, start_mid=2_230_000.0):
    """Generated windows (N, T, 41) -> order books (N, T, 40), by decoding each window."""
    return np.stack([decode(w, mean, std, start_mid) for w in windows])       # decode each window w

def invalid_rates(books):
    """Fraction of snapshots breaking each rule: (spread, ladder, volume)."""
    flat = books.reshape(-1, 40)                  # (N, T, 40) -> (N*T, 40): one row per snapshot
    n_spread, n_ladder, n_volume = check_rules(flat)            # rule checker
    rows = len(flat)
    return n_spread / rows, n_ladder / rows, n_volume / rows                                    # each count divided by the number of rows

def vwap_offset(books):
    """How far the whole book's VWAP sits from the mid, in ticks. One value per snapshot."""
    prices = np.concatenate([books[..., 0::4], books[..., 2::4]], axis=-1)   # all 20 prices
    sizes = np.concatenate([books[..., 1::4], books[..., 3::4]], axis=-1)    # all 20 sizes
    vwap = (prices * sizes).sum(axis=-1) / sizes.sum(axis=-1)     # sum of (prices * sizes) over the last axis, divided by sum of sizes
    mid = (books[..., 0] + books[..., 2]) / 2    
    return (vwap - mid) / 100

def vwap_stats(books):
    """Mean and standard deviation of the VWAP offset over all snapshots."""
    v = vwap_offset(books)
    return v.mean(), v.std()

def volatility_clustering(books):
    """Correlation between the size of one mid-price move and the next. Positive = big moves follow big moves."""
    mid = (books[..., 0] + books[..., 2]) / 2                                 # (best ask + best bid) / 2, for every step of every window
    moves = np.abs(np.diff(mid, axis=-1))     # size of each move, along time
    now = moves[:, :-1].ravel()               # every move except the last
    nxt = moves[:, 1:].ravel()                                 # every move except the first
    return np.corrcoef(now, nxt)[0, 1]

def collect(loader):
    """Every (window, label) in a loader -> windows (N, T, 41) and labels (N,) as numpy."""
    windows, labels = [], []
    for w, s in loader:
        windows.append(w)
        labels.append(s)
    return torch.cat(windows).numpy(), torch.cat(labels).numpy()

def per_stock_report(real_w, real_l, fake_w, fake_l, mean, std, tickers):
    """Print rule breaks, VWAP offset and volatility clustering for each stock, real vs fake."""
    for stock_id, ticker in enumerate(tickers):
        real_books = windows_to_books(real_w[real_l == stock_id], mean, std)
        fake_books = windows_to_books(fake_w[fake_l == stock_id], mean, std)                       # same, for the fake windows of this stock
        spread, ladder, volume = invalid_rates(fake_books)
        r_mean, r_std = vwap_stats(real_books)
        f_mean, f_std = vwap_stats(fake_books)
        print(f"{ticker} | broken ladder {ladder:.1%} | VWAP real {r_mean:.2f}±{r_std:.2f} "
              f"fake {f_mean:.2f}±{f_std:.2f} | clustering real {volatility_clustering(real_books):.3f} "
              f"fake {volatility_clustering(fake_books):.3f}")

# TODO: spreads and returns
#   goal: pull the two quantities the spec's KL targets are about out of a set of books

# TODO: kl_divergence
#   goal: one number for how different two histograms are (target <= 0.1)
#   think: both histograms must use the same bins - why?

# TODO: real-vs-real reference
#   goal: score real training-hour windows against the test hour with the same metrics
#   (the best a perfect generator could do - your review promises this)

# TODO: ssim on depth heatmaps (last - ask your tutor which real windows to compare against)
