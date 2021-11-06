import numpy as np
import pandas as pd
from aggregations import roll_weighted_mean


def cython_testing():
    minp = 0
    vals = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
    weights = np.array([1/3] * 3, dtype=np.float64)

    k = roll_weighted_mean(vals, weights, minp)
    p = np.array(k)
    print(p)

if __name__ == "__main__":
    cython_testing()
