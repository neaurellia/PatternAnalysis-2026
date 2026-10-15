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

if __name__ == "__main__":
    book_file = "data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    msg_file = "data/AMZN_2012-06-21_34200000_57600000_message_10.csv"

    times, book = load_lobster(book_file, msg_file)
    print("events:", len(times), "| book shape:", book.shape)

    n_spread, n_ladder, n_volume = check_rules(book)
    print("rule violations (spread, ladder, volume):", n_spread, n_ladder, n_volume)

    grid, snapshots = resample(times, book)
    print("snapshots:", len(snapshots))