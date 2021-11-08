from time import time
import numpy as np
import pandas as pd
from aggregations import roll_weighted_mean


def cython_testing(arr: np.ndarray, w: int):
    weights = np.array([1] * w, dtype=np.float64)
    start = time()
    r = roll_weighted_mean(arr, weights, w)
    end = time()
    print(end - start)

if __name__ == "__main__":
    arr = np.array(range(1000000), dtype=np.float64)
    w = 4
    cython_testing(arr, w)
