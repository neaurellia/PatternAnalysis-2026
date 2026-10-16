# utils.py - scoring generated order books against real ones
# (all of these work on DECODED books, not on normalised features)

# TODO: invalid book rate
#   goal: % of generated snapshots that break the spread / ladder / volume rules
#   think: you already wrote something that counts this

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
