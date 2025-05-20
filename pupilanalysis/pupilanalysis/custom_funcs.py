# custom_funcs.py

import numpy as np
import pandas as pd
from scipy.stats import zscore
from datamatrix import operations as ops

# %% count_valid_traces

def count_valid_traces(traces):
    """
    Count the number of valid traces (not all NaN).

    Parameters:
        traces: array-like, shape (n_trials, timepoints)

    Returns:
        valid_count: int, number of non-empty traces
        valid_traces: np.ndarray with only valid rows
    """
    traces = np.asarray(traces)
    mask = ~np.all(np.isnan(traces), axis=1)
    valid_traces = traces[mask]
    return len(valid_traces), valid_traces

# %%

def baseline_correction(dm, col='ptrace', baseline_len=10, window_len=50):
    mean_values = []
    start_indices = []

    for row in dm:
        signal = getattr(row, col)
        signal = signal[:window_len]  # first 50 values only
        found = False

        for i in range(window_len-baseline_len+1):  # up to index 40 (inclusive) so i:i+10 is within bounds
            window = signal[i:i+baseline_len]
            if np.all(~np.isnan(window)):
                mean_values.append(np.mean(window))
                start_indices.append(i)
                found = True
                break

        if not found:
            mean_values.append(np.nan)
            start_indices.append(np.nan)
    
    print(f"Number of set baselines: {len(mean_values) - np.sum(np.isnan(mean_values))}")
    print(f"Start index of baselines: {pd.value_counts(np.array(start_indices))}")
    
    return(mean_values, start_indices)

# %%

def bl_to_zscore(dm, bl_col='baseline'):
    z_scores = []

    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        bl = getattr(inf, bl_col)
        z = zscore(bl, nan_policy='omit')
        z_scores.append(z.flatten())

    z_scores = np.concatenate(z_scores)
    
    return(z_scores)