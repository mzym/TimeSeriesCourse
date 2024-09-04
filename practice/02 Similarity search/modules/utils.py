import numpy as np
import pandas as pd
import math
import random


def read_ts(file_path: str) -> np.ndarray:
    """
    Read time series

    Parameters
    ----------
    file_path: Path to file where time series data are stored
     
    Returns
    -------
    ts: time series data
    """

    ts = pd.read_csv(file_path, header=None, delim_whitespace=True)
    
    return ts.to_numpy()


def z_normalize(ts: np.ndarray) -> np.ndarray:
    """
    Calculate the z-normalized time series by subtracting the mean and
    dividing by the standard deviation along a given axis

    Parameters
    ----------
    ts: time series
    
    Returns
    -------
    norm_ts: z-normalized time series
    """

    norm_ts = (ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)

    return norm_ts


def sliding_window(ts: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """
    Extract subsequences from time series using sliding window

    Parameters
    ----------
    ts: time series
    window: size of the sliding window
    step: step of the sliding window

    Returns
    -------
    subs_matrix: matrix of subsequences
    """
    
    n = ts.shape[0]
    N = math.ceil((n-window+1)/step)

    subs_matrix = np.zeros((N, window))

    for i in range(N):
        start_idx = i*step
        end_idx = start_idx + window
        subs_matrix[i] = ts[start_idx:end_idx]

    return subs_matrix


def random_walk(n: int) -> np.ndarray:
    """
    Generate the time series based on Random Walk model

    Parameters
    ----------
    n: length of time series
    
    Returns
    -------
    random_walk_ts: generated time series
    """

    value = 0

    random_walk_ts = [value]
    directions = ["UP", "DOWN"]

    for i in range(1, n):
        # Randomly select either UP or DOWN
        step = random.choice(directions)

        # Move the object up or down
        if step == "UP":
            value += 1
        elif step == "DOWN":
            value -= 1

        random_walk_ts.append(value)

    return np.array(random_walk_ts)
