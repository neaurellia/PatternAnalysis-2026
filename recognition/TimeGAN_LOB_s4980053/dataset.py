import numpy as np
import pandas as pd

def load_lobster(book_path, msg_path):
    """Return times (N,) in seconds after midnight, and book (N, 40)."""
    # read both CSVs 
    book_df = pd.read_csv(book_path, header=None)
    msg_df = pd.read_csv(msg_path, header=None)
    # times = column 0 of the message file, as a numpy array
    times = msg_df[0].to_numpy()
    # book = the whole orderbook file, as a numpy array
    book = np.array(book_df)
    return times, book

def check_rules(book):
    """Return the number of rows breaking each rule: spread, ladder, volume."""
    ask_price, ask_size = book[:, 0::4], book[:, 1::4]   # every 4th column, starting at 0 / 1
    bid_price, bid_size = book[:, 2::4], book[:, 3::4]
    
    # spread: best bid (col 0) must be below best ask (col 0)
    bad_spread = bid_price[:, 0] >= ask_price[:, 0]
    n_spread = bad_spread.sum()

    # ladder (ask): each level must be more expensive than the one before
    ask_gaps = np.diff(ask_price, axis=1) # ask2-ask1, ask3-ask2
    bad_ask = (ask_gaps <= 0).any(axis=1)
    
    # ladder (bid): each level must be cheaper than the one before
    bid_gaps = np.diff(bid_price, axis=1) # ask2-ask1, ask3-ask2
    bad_bid = (bid_gaps >= 0).any(axis=1)

    n_ladder = (bad_ask | bad_bid).sum()
    
    # volume: rows where any size < 0
    bad_volume = (ask_size < 0).any(axis=1) | (bid_size < 0).any(axis=1)
    n_volume = bad_volume.sum()

    return n_spread, n_ladder, n_volume

def resample(times, book, dt=1.0):
    """Book state at the end of every dt seconds from 9:30 to 16:00."""
    grid = np.arange(9.5 * 3600 + dt, 16 * 3600, dt)
    # for each grid time, the index of the LAST event at or before it
    idx = np.searchsorted(times, grid, side="right") - 1
    # return grid, book[those indices]
    return grid, book[idx]
    # question to think about: why does this "carry forward" automatically in empty seconds? lowk help idk

def encode(snapshots):
    """Snapshots (T, 40) -> features (T-1, 41): return, 20 price offsets, 20 log sizes."""
    ask_price, ask_size = snapshots[:, 0::4], snapshots[:, 1::4]
    bid_price, bid_size = snapshots[:, 2::4], snapshots[:, 3::4]

    mid = (ask_price[:, 0] + bid_price[:, 0]) / 2
    log_ret = np.diff(np.log(mid))

    # drop the first row from mid, ask_price, bid_price, ask_size, bid_size
    mid = mid[1:]
    ask_price = ask_price[1:]
    bid_price = bid_price[1:]
    ask_size = ask_size[1:]
    bid_size = bid_size[1:]   

    # price offsets in ticks: how far each level sits from the mid
    # 1 tick = 1 cent = 100 in LOBSTER units
    ask_off = (ask_price - mid[:, None]) / 100
    bid_off = (bid_price - mid[:, None]) / 100

    # sizes: log squashes 1..34,000 into a small range
    log_sizes = np.log(np.concatenate([ask_size, bid_size], axis=1))

    # glue everything side by side: 1 + 10 + 10 + 20 = 41 columns
    return np.concatenate([log_ret[:, None], ask_off, bid_off, log_sizes], axis=1)

def split_by_time(grid, features):
    """Split features into train / val / test by time of day."""
    t = grid[1:]                      # times that line up with the feature rows

    train = features[t < 14 * 3600]
    val   = features[(t >= 14 * 3600) & (t < 15 * 3600)]
    test  = features[t >= 15 * 3600]
    return train, val, test

def normalise(train, val, test):
    """Scale each column to mean 0, std 1 using TRAINING statistics only."""
    mean = train.mean(axis=0)          # one mean per column → shape (41,)
    std = train.std(axis=0) + 1e-8     # + tiny number: avoids dividing by 0 if a column never changes

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std
    return train, val, test, mean, std

if __name__ == "__main__":
    book_file = "data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    msg_file = "data/AMZN_2012-06-21_34200000_57600000_message_10.csv"

    times, book = load_lobster(book_file, msg_file)
    print("events:", len(times), "| book shape:", book.shape)

    n_spread, n_ladder, n_volume = check_rules(book)
    print("rule violations (spread, ladder, volume):", n_spread, n_ladder, n_volume)

    grid, snapshots = resample(times, book)
    print("snapshots:", len(snapshots))

    features = encode(snapshots)
    print("features:", features.shape)
    print("ask offsets all positive:", (features[:, 1:11] > 0).all())
    print("bid offsets all negative:", (features[:, 11:21] < 0).all())

    train, val, test = split_by_time(grid, features)
    print("train / val / test rows:", len(train), len(val), len(test))

    train, val, test, mean, std = normalise(train, val, test)
    print("train mean ~0:", train.mean().round(3), "| train std ~1:", train.std().round(3))
    print("val mean:", val.mean().round(3), "| val std:", val.std().round(3))