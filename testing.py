from time import time
import numpy as np
import pandas as pd
from aggregations import roll_weighted_mean, recursive_sma


def existing(arr: np.ndarray, w: int):
    weights = np.array([1] * w, dtype=np.float64)
    start = time()
    r = roll_weighted_mean(arr, weights, w)
    end = time()
    print('ans:', np.array(r))
    print(end - start)

def recursive_version(arr: np.ndarray, w: int):
    start = time()
    r = recursive_sma(arr, w)
    end = time()
    print('ans:', np.array(r))
    print(end - start)

if __name__ == "__main__":
    arr = np.array(range(10000000), dtype=np.float64)
    w = 4
    existing(arr, w)
    recursive_version(arr, w)
