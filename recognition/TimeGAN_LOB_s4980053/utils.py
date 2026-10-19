# utils.py - scoring generated order books against real ones
# (all of these work on DECODED books, not on normalised features)

import numpy as np

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

# TODO: spreads and returns
#   goal: pull the two quantities the spec's KL targets are about out of a set of books

# TODO: kl_divergence
#   goal: one number for how different two histograms are (target <= 0.1)
#   think: both histograms must use the same bins - why?

# TODO: volatility clustering
#   goal: do big moves follow big moves? one number, compare real vs fake

# TODO: real-vs-real reference
#   goal: score real training-hour windows against the test hour with the same metrics
#   (the best a perfect generator could do - your review promises this)

# TODO: ssim on depth heatmaps (last - ask your tutor which real windows to compare against)
